// Fast streaming implementation of cache_compact_trace.py's reference cache.
//
// The executable deliberately has a narrow binary interface.  It consumes the
// 12-byte little-endian compact records on stdin, emits the same record format
// on stdout, and writes a JSON census to --stats.  Object bases/extents are
// supplied by the Python wrapper so this code never interprets a research
// plan or silently changes its semantic bindings.

#include <openssl/sha.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

#pragma pack(push, 1)
struct Record {
    std::uint16_t object_index;
    std::uint32_t object_offset;
    std::uint16_t kernel_ordinal;
    std::uint16_t bytes;
    std::uint8_t operation;
    std::uint8_t flags;
};
#pragma pack(pop)

static_assert(sizeof(Record) == 12);

#pragma pack(push, 1)
struct TimedRecord {
    Record request;
    std::uint64_t issue_ns;
};
#pragma pack(pop)

static_assert(sizeof(TimedRecord) == 20);

#pragma pack(push, 1)
struct AffineFallbackRecord {
    std::uint16_t period_index;
    std::uint32_t read_ordinal;
    Record record;
};
#pragma pack(pop)

static_assert(sizeof(AffineFallbackRecord) == 18);

constexpr std::uint8_t kRead = 0;
constexpr std::uint8_t kWrite = 1;
constexpr std::uint8_t kEvictFirst = 1;

struct ObjectRow {
    std::uint64_t base = 0;
    std::uint64_t extent = 0;
};

struct Counters {
    std::uint64_t input_requests = 0;
    std::uint64_t input_bytes = 0;
    std::uint64_t input_r_bytes = 0;
    std::uint64_t input_w_bytes = 0;
    std::uint64_t output_requests = 0;
    std::uint64_t output_bytes = 0;
    std::uint64_t output_r_requests = 0;
    std::uint64_t output_r_bytes = 0;
    std::uint64_t output_w_requests = 0;
    std::uint64_t output_w_bytes = 0;
    std::uint64_t reason_read_miss = 0;
    std::uint64_t reason_dirty_eviction = 0;
    std::uint64_t reason_write_allocate_fill = 0;
    std::uint64_t reason_write_through = 0;
    std::uint64_t reason_write_around = 0;
    std::uint64_t reason_final_dirty_drain = 0;
    std::uint64_t read_hits = 0;
    std::uint64_t read_misses = 0;
    std::uint64_t write_hits = 0;
    std::uint64_t write_misses = 0;
    std::uint64_t line_evictions = 0;
    std::uint64_t dirty_sector_writebacks = 0;
    std::uint64_t final_dirty_sector_writebacks = 0;
    std::uint64_t sass_evict_first_read_accesses = 0;
    std::uint64_t write_miss_allocations_without_fetch = 0;
};

Counters subtract_counters(const Counters& left, const Counters& right) {
    Counters result;
#define HBF_COUNTER_DELTA(name) result.name = left.name - right.name
    HBF_COUNTER_DELTA(input_requests);
    HBF_COUNTER_DELTA(input_bytes);
    HBF_COUNTER_DELTA(input_r_bytes);
    HBF_COUNTER_DELTA(input_w_bytes);
    HBF_COUNTER_DELTA(output_requests);
    HBF_COUNTER_DELTA(output_bytes);
    HBF_COUNTER_DELTA(output_r_requests);
    HBF_COUNTER_DELTA(output_r_bytes);
    HBF_COUNTER_DELTA(output_w_requests);
    HBF_COUNTER_DELTA(output_w_bytes);
    HBF_COUNTER_DELTA(reason_read_miss);
    HBF_COUNTER_DELTA(reason_dirty_eviction);
    HBF_COUNTER_DELTA(reason_write_allocate_fill);
    HBF_COUNTER_DELTA(reason_write_through);
    HBF_COUNTER_DELTA(reason_write_around);
    HBF_COUNTER_DELTA(reason_final_dirty_drain);
    HBF_COUNTER_DELTA(read_hits);
    HBF_COUNTER_DELTA(read_misses);
    HBF_COUNTER_DELTA(write_hits);
    HBF_COUNTER_DELTA(write_misses);
    HBF_COUNTER_DELTA(line_evictions);
    HBF_COUNTER_DELTA(dirty_sector_writebacks);
    HBF_COUNTER_DELTA(final_dirty_sector_writebacks);
    HBF_COUNTER_DELTA(sass_evict_first_read_accesses);
    HBF_COUNTER_DELTA(write_miss_allocations_without_fetch);
#undef HBF_COUNTER_DELTA
    return result;
}

class Sha256 {
public:
    Sha256() {
        if (SHA256_Init(&context_) != 1) {
            throw std::runtime_error("SHA256_Init failed");
        }
    }

    void update(const void* data, std::size_t bytes) {
        if (bytes != 0 && SHA256_Update(&context_, data, bytes) != 1) {
            throw std::runtime_error("SHA256_Update failed");
        }
    }

    std::string finish() {
        std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
        if (finished_ || SHA256_Final(digest.data(), &context_) != 1) {
            throw std::runtime_error("SHA256_Final failed");
        }
        finished_ = true;
        std::ostringstream stream;
        stream << std::hex << std::setfill('0');
        for (const auto value : digest) {
            stream << std::setw(2) << static_cast<unsigned>(value);
        }
        return stream.str();
    }

private:
    SHA256_CTX context_{};
    bool finished_ = false;
};

class StageOutputTracker {
public:
    StageOutputTracker() : digest_(std::make_unique<Sha256>()) {}

    void observe(const Record& record) {
        digest_->update(&record, sizeof(record));
        ++records_;
    }

    std::pair<std::string, std::uint64_t> finish_stage() {
        auto result = std::make_pair(digest_->finish(), records_);
        digest_ = std::make_unique<Sha256>();
        records_ = 0;
        return result;
    }

    std::uint64_t records() const { return records_; }

private:
    std::unique_ptr<Sha256> digest_;
    std::uint64_t records_ = 0;
};

struct AffinePeriodRow {
    std::uint64_t period_index = 0;
    std::uint64_t input_requests = 0;
    std::uint64_t output_requests = 0;
    std::uint64_t read_requests = 0;
    std::uint64_t write_requests = 0;
    std::string normalized_read_sha256;
};

class AffinePeriodAudit {
public:
    AffinePeriodAudit(
        std::uint64_t prefix_input_records,
        std::uint64_t input_records_per_period,
        std::vector<std::int64_t> object_period_deltas,
        std::string read_template_path = {},
        std::string writeback_fallback_path = {})
        : prefix_input_records_(prefix_input_records),
          input_records_per_period_(input_records_per_period),
          object_period_deltas_(std::move(object_period_deltas)),
          read_template_path_(std::move(read_template_path)),
          writeback_fallback_path_(std::move(writeback_fallback_path)),
          read_digest_(std::make_unique<Sha256>()),
          region_output_digest_(std::make_unique<Sha256>()) {
        if (input_records_per_period_ == 0 || object_period_deltas_.empty()) {
            throw std::runtime_error("invalid affine-period audit configuration");
        }
        if (read_template_path_.empty() != writeback_fallback_path_.empty()) {
            throw std::runtime_error(
                "affine-period template and fallback outputs must be paired");
        }
        if (!read_template_path_.empty()) {
            if (std::filesystem::exists(read_template_path_) ||
                std::filesystem::exists(writeback_fallback_path_)) {
                throw std::runtime_error(
                    "refusing to overwrite affine-period macro artifacts");
            }
            read_template_output_.open(read_template_path_, std::ios::binary);
            writeback_fallback_output_.open(
                writeback_fallback_path_, std::ios::binary);
            if (!read_template_output_ || !writeback_fallback_output_) {
                throw std::runtime_error(
                    "cannot open affine-period macro artifact output");
            }
            read_template_digest_ = std::make_unique<Sha256>();
            writeback_fallback_digest_ = std::make_unique<Sha256>();
        }
    }

    void observe(const Record& record, std::uint64_t input_requests) {
        if (input_requests <= prefix_input_records_) return;
        const auto region_input_requests = input_requests - prefix_input_records_;
        if ((region_input_requests - 1) / input_records_per_period_ != period_index_) {
            throw std::runtime_error("affine-period output escaped its input period");
        }
        region_output_digest_->update(&record, sizeof(record));
        ++region_output_requests_;
        ++output_requests_;
        if (record.operation != kRead) {
            if (writeback_fallback_output_.is_open()) {
                if (period_index_ > std::numeric_limits<std::uint16_t>::max() ||
                    read_requests_ > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error(
                        "affine-period fallback coordinate overflows");
                }
                const AffineFallbackRecord fallback{
                    .period_index = static_cast<std::uint16_t>(period_index_),
                    .read_ordinal = static_cast<std::uint32_t>(read_requests_),
                    .record = record,
                };
                writeback_fallback_output_.write(
                    reinterpret_cast<const char*>(&fallback), sizeof(fallback));
                if (!writeback_fallback_output_) {
                    throw std::runtime_error(
                        "failed to write affine-period fallback record");
                }
                writeback_fallback_digest_->update(&fallback, sizeof(fallback));
                ++writeback_fallback_records_;
            }
            ++region_write_requests_;
            ++write_requests_;
            return;
        }
        if (record.object_index >= object_period_deltas_.size()) {
            throw std::runtime_error("affine-period object index escapes delta table");
        }
        const auto delta = object_period_deltas_[record.object_index];
        const auto translated = static_cast<__int128>(delta) * period_index_;
        const auto normalized =
            static_cast<__int128>(record.object_offset) - translated;
        if (normalized < 0 ||
            normalized > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("affine-period normalized offset escapes u32");
        }
        auto canonical = record;
        canonical.object_offset = static_cast<std::uint32_t>(normalized);
        read_digest_->update(&canonical, sizeof(canonical));
        if (period_index_ == 0 && read_template_output_.is_open()) {
            read_template_output_.write(
                reinterpret_cast<const char*>(&canonical), sizeof(canonical));
            if (!read_template_output_) {
                throw std::runtime_error(
                    "failed to write affine-period read template");
            }
            read_template_digest_->update(&canonical, sizeof(canonical));
            ++read_template_records_;
        }
        ++region_read_requests_;
        ++read_requests_;
    }

    void finish_input_record(std::uint64_t input_requests) {
        if (input_requests <= prefix_input_records_) return;
        const auto region_input_requests = input_requests - prefix_input_records_;
        if (region_input_requests % input_records_per_period_ != 0) return;
        if (region_input_requests / input_records_per_period_ != period_index_ + 1) {
            throw std::runtime_error("affine-period input boundary is inconsistent");
        }
        rows_.push_back(AffinePeriodRow{
            .period_index = period_index_,
            .input_requests = input_records_per_period_,
            .output_requests = output_requests_,
            .read_requests = read_requests_,
            .write_requests = write_requests_,
            .normalized_read_sha256 = read_digest_->finish(),
        });
        ++period_index_;
        output_requests_ = 0;
        read_requests_ = 0;
        write_requests_ = 0;
        read_digest_ = std::make_unique<Sha256>();
    }

    void finish(std::uint64_t input_requests) {
        if (input_requests <= prefix_input_records_) {
            throw std::runtime_error(
                "affine-period audit has no records after its prefix");
        }
        const auto region_input_requests = input_requests - prefix_input_records_;
        if (region_input_requests % input_records_per_period_ != 0 ||
            rows_.size() != region_input_requests / input_records_per_period_ ||
            output_requests_ != 0 || read_requests_ != 0 || write_requests_ != 0) {
            throw std::runtime_error(
                "affine-period audit requires complete nonempty input periods");
        }
        region_output_sha256_ = region_output_digest_->finish();
        if (read_template_output_.is_open()) {
            read_template_output_.close();
            writeback_fallback_output_.close();
            if (!read_template_output_ || !writeback_fallback_output_) {
                throw std::runtime_error(
                    "failed to close affine-period macro artifacts");
            }
            read_template_sha256_ = read_template_digest_->finish();
            writeback_fallback_sha256_ = writeback_fallback_digest_->finish();
        }
    }

    std::uint64_t prefix_input_records() const { return prefix_input_records_; }
    std::uint64_t input_records_per_period() const {
        return input_records_per_period_;
    }
    std::uint64_t region_input_records() const {
        return rows_.size() * input_records_per_period_;
    }
    std::uint64_t region_output_requests() const {
        return region_output_requests_;
    }
    std::uint64_t region_read_requests() const { return region_read_requests_; }
    std::uint64_t region_write_requests() const { return region_write_requests_; }
    const std::string& region_output_sha256() const {
        return region_output_sha256_;
    }
    const std::vector<std::int64_t>& object_period_deltas() const {
        return object_period_deltas_;
    }
    const std::vector<AffinePeriodRow>& rows() const { return rows_; }
    bool writes_artifacts() const { return !read_template_path_.empty(); }
    const std::string& read_template_path() const { return read_template_path_; }
    const std::string& writeback_fallback_path() const {
        return writeback_fallback_path_;
    }
    std::uint64_t read_template_records() const { return read_template_records_; }
    std::uint64_t writeback_fallback_records() const {
        return writeback_fallback_records_;
    }
    const std::string& read_template_sha256() const {
        return read_template_sha256_;
    }
    const std::string& writeback_fallback_sha256() const {
        return writeback_fallback_sha256_;
    }

private:
    std::uint64_t prefix_input_records_ = 0;
    std::uint64_t input_records_per_period_ = 0;
    std::vector<std::int64_t> object_period_deltas_;
    std::string read_template_path_;
    std::string writeback_fallback_path_;
    std::uint64_t period_index_ = 0;
    std::uint64_t output_requests_ = 0;
    std::uint64_t read_requests_ = 0;
    std::uint64_t write_requests_ = 0;
    std::unique_ptr<Sha256> read_digest_;
    std::unique_ptr<Sha256> region_output_digest_;
    std::vector<AffinePeriodRow> rows_;
    std::ofstream read_template_output_;
    std::ofstream writeback_fallback_output_;
    std::unique_ptr<Sha256> read_template_digest_;
    std::unique_ptr<Sha256> writeback_fallback_digest_;
    std::uint64_t read_template_records_ = 0;
    std::uint64_t writeback_fallback_records_ = 0;
    std::string read_template_sha256_;
    std::string writeback_fallback_sha256_;
    std::uint64_t region_output_requests_ = 0;
    std::uint64_t region_read_requests_ = 0;
    std::uint64_t region_write_requests_ = 0;
    std::string region_output_sha256_;
};

class Output {
public:
    Output(
        FILE* stream,
        Counters& counters,
        Sha256& wire_digest,
        Sha256& untimed_digest,
        bool timed,
        AffinePeriodAudit* affine_period_audit = nullptr,
        StageOutputTracker* stage_output_tracker = nullptr)
        : stream_(stream),
          counters_(counters),
          wire_digest_(wire_digest),
          untimed_digest_(untimed_digest),
          timed_(timed),
          affine_period_audit_(affine_period_audit),
          stage_output_tracker_(stage_output_tracker) {
        records_.reserve(1U << 16);
        timed_records_.reserve(1U << 16);
    }

    ~Output() {
        try {
            flush();
        } catch (...) {
        }
    }

    void begin_access(std::uint64_t issue_ns) {
        if (timed_ && has_issue_ns_ && issue_ns < current_issue_ns_) {
            throw std::runtime_error("timed cache input issue_ns regressed");
        }
        current_issue_ns_ = issue_ns;
        has_issue_ns_ = true;
    }

    void emit(Record record, std::string_view reason) {
        record.flags = 0;
        if (stage_output_tracker_ != nullptr) {
            stage_output_tracker_->observe(record);
        }
        if (affine_period_audit_ != nullptr) {
            affine_period_audit_->observe(record, counters_.input_requests);
        }
        observe_run(record);
        untimed_digest_.update(&record, sizeof(record));
        if (timed_) {
            if (!has_issue_ns_) {
                throw std::runtime_error(
                    "timed cache output has no triggering issue_ns");
            }
            const TimedRecord timed_record{
                .request = record,
                .issue_ns = current_issue_ns_,
            };
            timed_records_.push_back(timed_record);
            wire_digest_.update(&timed_record, sizeof(timed_record));
        } else {
            records_.push_back(record);
            wire_digest_.update(&record, sizeof(record));
        }
        ++counters_.output_requests;
        counters_.output_bytes += record.bytes;
        if (record.operation == kRead) {
            ++counters_.output_r_requests;
            counters_.output_r_bytes += record.bytes;
        } else {
            ++counters_.output_w_requests;
            counters_.output_w_bytes += record.bytes;
        }
        if (reason == "read-miss") ++counters_.reason_read_miss;
        else if (reason == "dirty-eviction") ++counters_.reason_dirty_eviction;
        else if (reason == "write-allocate-fill") ++counters_.reason_write_allocate_fill;
        else if (reason == "write-through") ++counters_.reason_write_through;
        else if (reason == "write-around") ++counters_.reason_write_around;
        else if (reason == "final-dirty-drain") ++counters_.reason_final_dirty_drain;
        else throw std::runtime_error("unknown output reason");
        if ((!timed_ && records_.size() == records_.capacity()) ||
            (timed_ && timed_records_.size() == timed_records_.capacity())) {
            flush();
        }
    }

    void flush() {
        if (timed_) {
            if (timed_records_.empty()) return;
            const auto written = std::fwrite(
                timed_records_.data(), sizeof(TimedRecord),
                timed_records_.size(), stream_);
            if (written != timed_records_.size()) {
                throw std::runtime_error(
                    "failed to write timed compact cache output");
            }
            timed_records_.clear();
        } else {
            if (records_.empty()) return;
            const auto written = std::fwrite(
                records_.data(), sizeof(Record), records_.size(), stream_);
            if (written != records_.size()) {
                throw std::runtime_error("failed to write compact cache output");
            }
            records_.clear();
        }
    }

    void finish_runs() {
        if (current_run_length_ != 0) {
            ++run_histogram_[current_run_length_];
            ++run_count_;
            maximum_run_length_ = std::max(
                maximum_run_length_, current_run_length_);
            current_run_length_ = 0;
        }
        if (run_records_ != counters_.output_requests) {
            throw std::runtime_error(
                "output contiguous-run census does not conserve records");
        }
    }

    std::uint64_t run_records() const { return run_records_; }
    std::uint64_t run_count() const { return run_count_; }
    std::uint64_t maximum_run_length() const { return maximum_run_length_; }
    const std::map<std::uint64_t, std::uint64_t>& run_histogram() const {
        return run_histogram_;
    }

private:
    static bool continues(const Record& left, const Record& right) {
        return left.object_index == right.object_index &&
            left.kernel_ordinal == right.kernel_ordinal &&
            left.bytes == right.bytes &&
            left.operation == right.operation &&
            left.flags == right.flags &&
            static_cast<std::uint64_t>(right.object_offset) ==
                static_cast<std::uint64_t>(left.object_offset) + left.bytes;
    }

    void observe_run(const Record& record) {
        if (!has_previous_ || !continues(previous_, record)) {
            if (current_run_length_ != 0) {
                ++run_histogram_[current_run_length_];
                ++run_count_;
                maximum_run_length_ = std::max(
                    maximum_run_length_, current_run_length_);
            }
            current_run_length_ = 1;
        } else {
            ++current_run_length_;
        }
        previous_ = record;
        has_previous_ = true;
        ++run_records_;
    }

    FILE* stream_;
    Counters& counters_;
    Sha256& wire_digest_;
    Sha256& untimed_digest_;
    bool timed_ = false;
    AffinePeriodAudit* affine_period_audit_ = nullptr;
    StageOutputTracker* stage_output_tracker_ = nullptr;
    std::vector<Record> records_;
    std::vector<TimedRecord> timed_records_;
    std::uint64_t current_issue_ns_ = 0;
    bool has_issue_ns_ = false;
    Record previous_{};
    bool has_previous_ = false;
    std::uint64_t current_run_length_ = 0;
    std::uint64_t run_records_ = 0;
    std::uint64_t run_count_ = 0;
    std::uint64_t maximum_run_length_ = 0;
    std::map<std::uint64_t, std::uint64_t> run_histogram_;
};

struct Entry {
    std::uint64_t tag = 0;
    std::uint64_t present = 0;
    std::uint64_t dirty = 0;
    bool valid = false;
};

#pragma pack(push, 1)
struct SemanticSectorState {
    std::uint16_t object_index;
    std::uint32_t sector_object_offset;
    std::uint8_t dirty;
    std::uint16_t writer_kernel_ordinal;
    std::uint8_t writer_operation;
    std::uint32_t writer_object_offset;
    std::uint16_t writer_bytes;
};
#pragma pack(pop)

static_assert(sizeof(SemanticSectorState) == 16);

class Cache {
public:
    Cache(
        std::vector<ObjectRow> objects,
        std::uint64_t capacity_bytes,
        std::uint64_t line_bytes,
        std::uint64_t sector_bytes,
        std::uint64_t associativity,
        bool write_back,
        bool write_allocate,
        bool write_miss_fetch,
        Counters& counters,
        Output& output,
        std::uint64_t read_fill_bytes = 0,
        bool clip_read_fill_to_object = false,
        std::uint64_t writeback_bytes = 0)
        : objects_(std::move(objects)),
          line_bytes_(line_bytes),
          sector_bytes_(sector_bytes),
          associativity_(associativity),
          write_back_(write_back),
          write_allocate_(write_allocate),
          write_miss_fetch_(write_miss_fetch),
          counters_(counters),
          output_(output),
          read_fill_bytes_(read_fill_bytes),
          clip_read_fill_to_object_(clip_read_fill_to_object),
          writeback_bytes_(writeback_bytes) {
        if (capacity_bytes == 0 || line_bytes_ == 0 || sector_bytes_ == 0 ||
            associativity_ == 0 || line_bytes_ % sector_bytes_ != 0) {
            throw std::runtime_error("invalid cache geometry");
        }
        sectors_per_line_ = line_bytes_ / sector_bytes_;
        // Opt-in functional fill-policy experiment, not a change to the
        // legacy sector-demand profile. No memory completion/MSHR timing is
        // modeled: a fill is visible to the next ordered input request.
        if (read_fill_bytes_ != 0) {
            if (read_fill_bytes_ != 64 || sector_bytes_ != 32 ||
                line_bytes_ % read_fill_bytes_ != 0) {
                throw std::runtime_error(
                    "read-fill experiment requires 64B fills, 32B sectors and 64B-divisible lines");
            }
            std::vector<std::pair<std::uint64_t, std::uint64_t>> extents;
            for (const auto& object : objects_) {
                if (object.base % sector_bytes_ != 0 ||
                    object.base + object.extent >
                        std::numeric_limits<std::uint64_t>::max() - read_fill_bytes_) {
                    throw std::runtime_error("read-fill object is unaligned or overflows");
                }
                const auto end = object.base + object.extent;
                extents.emplace_back(object.base,
                    ((end + sector_bytes_ - 1) / sector_bytes_) * sector_bytes_);
            }
            std::sort(extents.begin(), extents.end());
            for (std::size_t i = 1; i < extents.size(); ++i) {
                if (extents[i].first < extents[i - 1].second) {
                    throw std::runtime_error("read-fill objects share a sector");
                }
            }
        } else if (clip_read_fill_to_object_) {
            throw std::runtime_error("read-fill object clipping requires an explicit fill size");
        }
        if (writeback_bytes_ != 0 &&
            (writeback_bytes_ != 64 || read_fill_bytes_ != 64 ||
             !write_back_ || !write_allocate_)) {
            throw std::runtime_error(
                "64B writeback requires 64B read fills and write-back/write-allocate");
        }
        if (sectors_per_line_ > 64) {
            throw std::runtime_error("fast cache supports at most 64 sectors per line");
        }
        const auto lines = capacity_bytes / line_bytes_;
        if (lines < associativity_ || lines % associativity_ != 0 ||
            associativity_ > std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error("cache capacity/associativity is incompatible");
        }
        set_count_ = lines / associativity_;
        if (set_count_ > std::numeric_limits<std::size_t>::max() / associativity_) {
            throw std::runtime_error("cache geometry is too large");
        }
        const auto entries = static_cast<std::size_t>(set_count_ * associativity_);
        entries_.resize(entries);
        order_.resize(entries);
        used_.assign(static_cast<std::size_t>(set_count_), 0);
        dirty_records_.resize(entries * static_cast<std::size_t>(sectors_per_line_));
        dirty_order_.assign(entries * static_cast<std::size_t>(sectors_per_line_), 0xff);
    }

    void access(const Record& request, std::uint64_t issue_ns = 0) {
        validate(request);
        output_.begin_access(issue_ns);
        last_kernel_ = request.kernel_ordinal;
        ++counters_.input_requests;
        counters_.input_bytes += request.bytes;
        if (request.operation == kRead) counters_.input_r_bytes += request.bytes;
        else counters_.input_w_bytes += request.bytes;

        const auto address = objects_[request.object_index].base + request.object_offset;
        const auto line_number = address / line_bytes_;
        const auto set = line_number % set_count_;
        const auto tag = line_number / set_count_;
        const auto sector = (address % line_bytes_) / sector_bytes_;
        const auto bit = std::uint64_t{1} << sector;
        const auto found = find(set, tag);
        const bool hit = found >= 0 && (entry(set, found).present & bit) != 0;
        const bool evict_first = request.operation == kRead &&
            (request.flags & kEvictFirst) != 0;
        if (evict_first) ++counters_.sass_evict_first_read_accesses;

        if (request.operation == kRead) {
            if (hit) {
                ++counters_.read_hits;
                touch(set, static_cast<std::uint16_t>(found), evict_first);
                return;
            }
            ++counters_.read_misses;
            auto way = found;
            if (way < 0) way = insert(set, tag, request.kernel_ordinal);
            else touch(set, static_cast<std::uint16_t>(way), false);
            auto& value = entry(set, way);
            if (read_fill_bytes_ != 0) {
                fill_read(request, value, "read-miss");
            } else {
                value.present |= bit;
                auto downstream = request;
                downstream.operation = kRead;
                output_.emit(downstream, "read-miss");
            }
            touch(set, static_cast<std::uint16_t>(way), evict_first);
            return;
        }

        if (hit) ++counters_.write_hits;
        else ++counters_.write_misses;
        if (!write_back_) {
            if (hit) {
                touch(set, static_cast<std::uint16_t>(found), false);
            } else if (write_allocate_) {
                auto way = found;
                if (way < 0) way = insert(set, tag, request.kernel_ordinal);
                else touch(set, static_cast<std::uint16_t>(way), false);
                entry(set, way).present |= bit;
            }
            auto downstream = request;
            downstream.operation = kWrite;
            output_.emit(downstream, "write-through");
            return;
        }

        if (!hit && !write_allocate_) {
            auto downstream = request;
            downstream.operation = kWrite;
            output_.emit(downstream, "write-around");
            return;
        }
        auto way = found;
        if (!hit) {
            if (way < 0) way = insert(set, tag, request.kernel_ordinal);
            else touch(set, static_cast<std::uint16_t>(way), false);
            if (write_miss_fetch_) {
                if (read_fill_bytes_ != 0) {
                    fill_read(request, entry(set, way), "write-allocate-fill");
                } else {
                    auto fill = request;
                    fill.operation = kRead;
                    output_.emit(fill, "write-allocate-fill");
                }
            } else {
                ++counters_.write_miss_allocations_without_fetch;
            }
        }
        auto& value = entry(set, way);
        value.present |= bit;
        if ((value.dirty & bit) == 0) {
            dirty_order_[dirty_index(set, way, sector)] = dirty_count(value);
        }
        value.dirty |= bit;
        dirty_records_[dirty_index(set, way, sector)] = request;
        touch(set, static_cast<std::uint16_t>(way), false);
    }

    void drain() {
        for (std::uint64_t set = 0; set < set_count_; ++set) {
            const auto used = used_[static_cast<std::size_t>(set)];
            for (std::uint16_t position = 0; position < used; ++position) {
                const auto way = order_[order_index(set, position)];
                auto& value = entry(set, way);
                const auto count = emit_dirty(
                    set, way, value, "final-dirty-drain", last_kernel_);
                counters_.final_dirty_sector_writebacks += count;
            }
        }
    }

    std::uint64_t resident_lines() const {
        std::uint64_t total = 0;
        for (const auto value : used_) total += value;
        return total;
    }

    std::uint64_t resident_sectors() const {
        std::uint64_t total = 0;
        for (const auto& value : entries_) total += popcount(value.present);
        return total;
    }

    std::uint64_t resident_dirty_sectors() const {
        std::uint64_t total = 0;
        for (const auto& value : entries_) total += popcount(value.dirty);
        return total;
    }

    void write_read_fill_json(std::ostream& out) const {
        if (read_fill_bytes_ == 0) return;
        out << "  \"read_fill\": {\"bytes\": " << read_fill_bytes_
            << ",\"applies_to\":\"load miss and enabled write-allocate read fetch\""
            << ",\"writeback_policy\":\""
            << (writeback_bytes_ ? "valid-owned-64B-groups" : "unchanged dirty-sector writes") << "\""
            << ",\"visibility\":\"functional immediate fill in input order; no MSHR/completion timing\""
            << ",\"object_edge_policy\":\""
            << (clip_read_fill_to_object_ ? "clip-explicitly" : "reject") << "\""
            << ",\"fill_requests\":" << read_fill_requests_
            << ",\"fill_bytes\":" << read_fill_output_bytes_
            << ",\"additional_clean_sectors_installed\":" << additional_fill_sectors_
            << ",\"object_edge_clipped_requests\":" << clipped_fill_requests_
            << ",\"object_edge_unrepresented_bytes\":" << clipped_fill_bytes_
            << "},\n";
        if (writeback_bytes_ != 0) {
            out << "  \"writeback\": {\"bytes\":64"
                << ",\"policy\":\"one owned, fully valid 64B group per eviction/drain; first-dirty group order\""
                << ",\"missing_sibling_policy\":\"reject; never synthesize a read or unknown data\""
                << ",\"object_edge_policy\":\"reject\""
                << ",\"requests\":" << grouped_writebacks_
                << ",\"output_bytes\":" << grouped_writebacks_ * 64
                << ",\"covered_dirty_sectors\":" << grouped_dirty_sectors_
                << ",\"legacy_dirty_record_bytes\":" << grouped_legacy_dirty_bytes_
                << ",\"extra_clean_or_sector_padding_bytes\":"
                << grouped_writebacks_ * 64 - grouped_legacy_dirty_bytes_
                << ",\"two_dirty_sector_groups\":" << two_dirty_sector_groups_
                << ",\"one_dirty_sector_groups\":" << one_dirty_sector_groups_
                << ",\"additional_read_requests\":0},\n";
        }
    }

    std::string physical_mru_to_lru_sha256() const {
        Sha256 digest;
        for (std::uint64_t set = 0; set < set_count_; ++set) {
            digest.update(&set, sizeof(set));
            const auto used = used_[static_cast<std::size_t>(set)];
            digest.update(&used, sizeof(used));
            for (std::uint16_t reverse = 0; reverse < used; ++reverse) {
                const auto position = static_cast<std::uint16_t>(used - 1 - reverse);
                digest.update(&position, sizeof(position));
                const auto way = order_[order_index(set, position)];
                const auto& value = entry(set, way);
                digest.update(&value.tag, sizeof(value.tag));
                digest.update(&value.present, sizeof(value.present));
                digest.update(&value.dirty, sizeof(value.dirty));
                for (std::uint64_t sector = 0; sector < sectors_per_line_; ++sector) {
                    const auto index = dirty_index(set, way, sector);
                    if ((value.dirty & (std::uint64_t{1} << sector)) != 0) {
                        digest.update(&dirty_records_[index], sizeof(Record));
                        digest.update(&dirty_order_[index], sizeof(dirty_order_[index]));
                    }
                }
            }
        }
        return digest.finish();
    }

    std::string semantic_contents_sha256() const {
        if (objects_.size() > std::numeric_limits<std::uint16_t>::max()) {
            throw std::runtime_error("semantic state has too many objects");
        }
        std::vector<SemanticSectorState> sectors;
        sectors.reserve(static_cast<std::size_t>(resident_sectors()));
        for (std::uint64_t set = 0; set < set_count_; ++set) {
            const auto used = used_[static_cast<std::size_t>(set)];
            for (std::uint16_t way = 0; way < used; ++way) {
                const auto& value = entry(set, way);
                if (!value.valid) continue;
                const auto line_number = value.tag * set_count_ + set;
                const auto line_address = line_number * line_bytes_;
                for (std::uint64_t sector = 0; sector < sectors_per_line_; ++sector) {
                    const auto bit = std::uint64_t{1} << sector;
                    if ((value.present & bit) == 0) continue;
                    const auto address = line_address + sector * sector_bytes_;
                    const auto [object_index, object_offset] = locate_object(address);
                    const bool dirty = (value.dirty & bit) != 0;
                    SemanticSectorState row{
                        .object_index = object_index,
                        .sector_object_offset = object_offset,
                        .dirty = static_cast<std::uint8_t>(dirty),
                        .writer_kernel_ordinal = 0,
                        .writer_operation = 0,
                        .writer_object_offset = 0,
                        .writer_bytes = 0,
                    };
                    if (dirty) {
                        const auto& record = dirty_records_[
                            dirty_index(set, way, sector)];
                        if (record.object_index != object_index) {
                            throw std::runtime_error(
                                "dirty record object differs from resident semantic sector");
                        }
                        row.writer_kernel_ordinal = record.kernel_ordinal;
                        row.writer_operation = record.operation;
                        row.writer_object_offset = record.object_offset;
                        row.writer_bytes = record.bytes;
                    }
                    sectors.push_back(row);
                }
            }
        }
        std::sort(
            sectors.begin(), sectors.end(),
            [](const auto& left, const auto& right) {
                return std::tie(
                    left.object_index, left.sector_object_offset, left.dirty,
                    left.writer_kernel_ordinal, left.writer_operation,
                    left.writer_object_offset, left.writer_bytes)
                    < std::tie(
                    right.object_index, right.sector_object_offset, right.dirty,
                    right.writer_kernel_ordinal, right.writer_operation,
                    right.writer_object_offset, right.writer_bytes);
            });
        Sha256 digest;
        digest.update(sectors.data(), sectors.size() * sizeof(SemanticSectorState));
        return digest.finish();
    }

private:
    void fill_read(const Record& request, Entry& value, std::string_view reason) {
        const auto& object = objects_[request.object_index];
        const auto address = object.base + request.object_offset;
        const auto aligned = (address / read_fill_bytes_) * read_fill_bytes_;
        const auto end = aligned + read_fill_bytes_;
        const auto begin_owned = std::max(aligned, object.base);
        const auto end_owned = std::min(end, object.base + object.extent);
        const auto bytes = end_owned - begin_owned;
        if (bytes != read_fill_bytes_) {
            if (!clip_read_fill_to_object_) {
                throw std::runtime_error("64B read fill escapes object; supply an owned allocation envelope or explicit clipping");
            }
            ++clipped_fill_requests_;
            clipped_fill_bytes_ += read_fill_bytes_ - bytes;
        }
        const auto offset = begin_owned - object.base;
        if (offset > std::numeric_limits<std::uint32_t>::max()) {
            throw std::runtime_error("read fill offset exceeds compact u32");
        }
        const auto first = (begin_owned % line_bytes_) / sector_bytes_;
        const auto last = ((end_owned - 1) % line_bytes_) / sector_bytes_;
        std::uint64_t fill_mask = 0;
        for (auto sector = first; sector <= last; ++sector) {
            fill_mask |= std::uint64_t{1} << sector;
        }
        const auto newly_present = popcount(fill_mask & ~value.present);
        if (newly_present == 0) throw std::runtime_error("read miss fills no absent sector");
        additional_fill_sectors_ += newly_present - 1;
        // Existing valid/dirty sibling data must survive the fetch. The model
        // tracks presence and dirtiness, not byte values from device memory.
        value.present |= fill_mask;
        auto downstream = request;
        downstream.object_offset = static_cast<std::uint32_t>(offset);
        downstream.bytes = static_cast<std::uint16_t>(bytes);
        downstream.operation = kRead;
        output_.emit(downstream, reason);
        ++read_fill_requests_;
        read_fill_output_bytes_ += bytes;
    }

    std::pair<std::uint16_t, std::uint32_t> locate_object(
        std::uint64_t address) const {
        for (std::size_t index = 0; index < objects_.size(); ++index) {
            const auto& object = objects_[index];
            if (address < object.base || address >= object.base + object.extent) {
                continue;
            }
            const auto offset = address - object.base;
            if (offset > std::numeric_limits<std::uint32_t>::max()) {
                throw std::runtime_error("semantic cache offset exceeds u32");
            }
            return {
                static_cast<std::uint16_t>(index),
                static_cast<std::uint32_t>(offset),
            };
        }
        throw std::runtime_error("resident cache sector is outside every object");
    }

    void validate(const Record& request) const {
        if (request.object_index >= objects_.size()) {
            throw std::runtime_error("compact request object index escapes table");
        }
        const auto& object = objects_[request.object_index];
        if (request.bytes == 0 ||
            static_cast<std::uint64_t>(request.object_offset) + request.bytes > object.extent) {
            throw std::runtime_error("compact request escapes object extent");
        }
        if (request.operation != kRead && request.operation != kWrite) {
            throw std::runtime_error("compact request has unsupported operation");
        }
        if ((request.flags & ~kEvictFirst) != 0 ||
            ((request.flags & kEvictFirst) != 0 && request.operation != kRead)) {
            throw std::runtime_error("compact request has unsupported flags");
        }
        const auto address = object.base + request.object_offset;
        if (address > std::numeric_limits<std::uint64_t>::max() - request.bytes) {
            throw std::runtime_error("compact request address overflows");
        }
        if (read_fill_bytes_ != 0 &&
            (address % sector_bytes_) + request.bytes > sector_bytes_) {
            throw std::runtime_error("read-fill input must be a single sector demand, not post-cache output");
        }
        if (writeback_bytes_ != 0 && !write_miss_fetch_ &&
            request.operation == kWrite &&
            (address % sector_bytes_ != 0 || request.bytes != sector_bytes_)) {
            throw std::runtime_error(
                "64B writeback cannot prove partial-sector validity without write-miss fetch");
        }
    }

    static std::uint8_t popcount(std::uint64_t value) {
        return static_cast<std::uint8_t>(__builtin_popcountll(value));
    }

    std::uint8_t dirty_count(const Entry& value) const {
        return popcount(value.dirty);
    }

    std::size_t entry_index(std::uint64_t set, std::int64_t way) const {
        return static_cast<std::size_t>(set * associativity_ + way);
    }

    std::size_t order_index(std::uint64_t set, std::uint16_t position) const {
        return static_cast<std::size_t>(set * associativity_ + position);
    }

    std::size_t dirty_index(
        std::uint64_t set, std::int64_t way, std::uint64_t sector) const {
        return entry_index(set, way) * static_cast<std::size_t>(sectors_per_line_) +
            static_cast<std::size_t>(sector);
    }

    Entry& entry(std::uint64_t set, std::int64_t way) {
        return entries_[entry_index(set, way)];
    }

    const Entry& entry(std::uint64_t set, std::int64_t way) const {
        return entries_[entry_index(set, way)];
    }

    std::int64_t find(std::uint64_t set, std::uint64_t tag) const {
        const auto used = used_[static_cast<std::size_t>(set)];
        for (std::uint16_t way = 0; way < used; ++way) {
            const auto& value = entry(set, way);
            if (value.valid && value.tag == tag) return way;
        }
        return -1;
    }

    void touch(std::uint64_t set, std::uint16_t way, bool low_priority) {
        const auto used = used_[static_cast<std::size_t>(set)];
        std::uint16_t position = 0;
        while (position < used && order_[order_index(set, position)] != way) {
            ++position;
        }
        if (position == used) throw std::runtime_error("cache LRU order lost a way");
        if (low_priority) {
            for (; position > 0; --position) {
                order_[order_index(set, position)] =
                    order_[order_index(set, position - 1)];
            }
            order_[order_index(set, 0)] = way;
        } else {
            for (; position + 1 < used; ++position) {
                order_[order_index(set, position)] =
                    order_[order_index(set, position + 1)];
            }
            order_[order_index(set, used - 1)] = way;
        }
    }

    std::int64_t insert(
        std::uint64_t set,
        std::uint64_t tag,
        std::uint16_t eviction_kernel) {
        auto& used = used_[static_cast<std::size_t>(set)];
        std::uint16_t way = 0;
        if (used < associativity_) {
            way = used;
            order_[order_index(set, used)] = way;
            ++used;
        } else {
            way = order_[order_index(set, 0)];
            auto& victim = entry(set, way);
            ++counters_.line_evictions;
            const auto dirty = emit_dirty(
                set, way, victim, "dirty-eviction", eviction_kernel);
            counters_.dirty_sector_writebacks += dirty;
            for (std::uint16_t position = 0; position + 1 < used; ++position) {
                order_[order_index(set, position)] =
                    order_[order_index(set, position + 1)];
            }
            order_[order_index(set, used - 1)] = way;
        }
        auto& value = entry(set, way);
        value = Entry{.tag = tag, .present = 0, .dirty = 0, .valid = true};
        const auto begin = dirty_index(set, way, 0);
        std::fill_n(
            dirty_order_.begin() + static_cast<std::ptrdiff_t>(begin),
            static_cast<std::size_t>(sectors_per_line_),
            static_cast<std::uint8_t>(0xff));
        return way;
    }

    std::uint64_t emit_dirty(
        std::uint64_t set,
        std::int64_t way,
        Entry& value,
        std::string_view reason,
        std::uint16_t emission_kernel) {
        const auto count = dirty_count(value);
        std::uint64_t emitted_groups = 0;
        for (std::uint8_t sequence = 0; sequence < count; ++sequence) {
            bool emitted = false;
            for (std::uint64_t sector = 0; sector < sectors_per_line_; ++sector) {
                const auto index = dirty_index(set, way, sector);
                if (dirty_order_[index] != sequence) continue;
                auto record = dirty_records_[index];
                record.operation = kWrite;
                record.kernel_ordinal = emission_kernel;
                if (writeback_bytes_ == 0) {
                    output_.emit(record, reason);
                } else {
                    const auto group = sector / 2;
                    const auto group_bit = std::uint64_t{1} << group;
                    if ((emitted_groups & group_bit) == 0) {
                        const auto first_sector = group * 2;
                        const auto mask = std::uint64_t{3} << first_sector;
                        if ((value.present & mask) != mask) {
                            throw std::runtime_error(
                                "64B writeback lacks a valid sibling sector; no implicit refill permitted");
                        }
                        const auto line_address =
                            (value.tag * set_count_ + set) * line_bytes_;
                        const auto address = line_address + first_sector * sector_bytes_;
                        const auto& object = objects_[record.object_index];
                        if (address < object.base ||
                            address + 64 > object.base + object.extent) {
                            throw std::runtime_error(
                                "64B writeback escapes owned object; no padding/clipping permitted");
                        }
                        const auto offset = address - object.base;
                        if (offset > std::numeric_limits<std::uint32_t>::max()) {
                            throw std::runtime_error("64B writeback offset exceeds compact u32");
                        }
                        const auto dirty_sectors = popcount(value.dirty & mask);
                        std::uint64_t legacy_bytes = 0;
                        for (auto s = first_sector; s < first_sector + 2; ++s) {
                            if ((value.dirty & (std::uint64_t{1} << s)) == 0) continue;
                            const auto& writer = dirty_records_[dirty_index(set, way, s)];
                            if (writer.object_index != record.object_index) {
                                throw std::runtime_error("64B writeback group crosses objects");
                            }
                            legacy_bytes += writer.bytes;
                        }
                        record.object_offset = static_cast<std::uint32_t>(offset);
                        record.bytes = 64;
                        output_.emit(record, reason);
                        emitted_groups |= group_bit;
                        ++grouped_writebacks_;
                        grouped_dirty_sectors_ += dirty_sectors;
                        grouped_legacy_dirty_bytes_ += legacy_bytes;
                        if (dirty_sectors == 2) ++two_dirty_sector_groups_;
                        else if (dirty_sectors == 1) ++one_dirty_sector_groups_;
                        else throw std::runtime_error("64B writeback has no dirty sector");
                    }
                }
                dirty_order_[index] = 0xff;
                emitted = true;
                break;
            }
            if (!emitted) throw std::runtime_error("cache dirty order is inconsistent");
        }
        value.dirty = 0;
        return count;
    }

    std::vector<ObjectRow> objects_;
    std::uint64_t line_bytes_ = 0;
    std::uint64_t sector_bytes_ = 0;
    std::uint64_t associativity_ = 0;
    std::uint64_t sectors_per_line_ = 0;
    std::uint64_t set_count_ = 0;
    bool write_back_ = true;
    bool write_allocate_ = true;
    bool write_miss_fetch_ = true;
    Counters& counters_;
    Output& output_;
    std::uint64_t read_fill_bytes_ = 0;
    bool clip_read_fill_to_object_ = false;
    std::uint64_t read_fill_requests_ = 0;
    std::uint64_t read_fill_output_bytes_ = 0;
    std::uint64_t additional_fill_sectors_ = 0;
    std::uint64_t clipped_fill_requests_ = 0;
    std::uint64_t clipped_fill_bytes_ = 0;
    std::uint64_t writeback_bytes_ = 0;
    std::uint64_t grouped_writebacks_ = 0;
    std::uint64_t grouped_dirty_sectors_ = 0;
    std::uint64_t grouped_legacy_dirty_bytes_ = 0;
    std::uint64_t two_dirty_sector_groups_ = 0;
    std::uint64_t one_dirty_sector_groups_ = 0;
    std::vector<Entry> entries_;
    std::vector<std::uint16_t> order_;
    std::vector<std::uint16_t> used_;
    std::vector<Record> dirty_records_;
    std::vector<std::uint8_t> dirty_order_;
    std::uint16_t last_kernel_ = 0;
};

struct StageBoundary {
    std::string label;
    std::uint64_t end_input_records = 0;
};

struct StageRow {
    std::string label;
    std::uint64_t begin_input_records = 0;
    std::uint64_t end_input_records = 0;
    Counters delta;
    std::string output_sha256;
    std::uint64_t output_digest_records = 0;
    std::uint64_t resident_lines = 0;
    std::uint64_t resident_sector_bytes = 0;
    std::uint64_t resident_dirty_sector_bytes = 0;
    std::string physical_mru_to_lru_sha256;
    std::string semantic_contents_sha256;
};

std::uint64_t parse_u64(const std::string& raw, const char* name) {
    std::size_t consumed = 0;
    const auto value = std::stoull(raw, &consumed, 10);
    if (consumed != raw.size()) throw std::runtime_error(std::string("invalid ") + name);
    return value;
}

std::int64_t parse_i64(const std::string& raw, const char* name) {
    std::size_t consumed = 0;
    const auto value = std::stoll(raw, &consumed, 10);
    if (consumed != raw.size()) throw std::runtime_error(std::string("invalid ") + name);
    return value;
}

bool parse_bool(const std::string& raw, const char* name) {
    if (raw == "1" || raw == "true") return true;
    if (raw == "0" || raw == "false") return false;
    throw std::runtime_error(std::string("invalid ") + name);
}

std::vector<StageBoundary> read_stage_boundaries(const std::string& path) {
    if (path.empty()) return {};
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open stage boundary table");
    std::vector<StageBoundary> result;
    StageBoundary row;
    while (stream >> row.label >> row.end_input_records) {
        if (row.label.empty() || row.end_input_records == 0 ||
            (!result.empty() &&
             row.end_input_records <= result.back().end_input_records)) {
            throw std::runtime_error(
                "stage boundaries must have nonempty labels and strictly increasing ends");
        }
        result.push_back(row);
    }
    if (!stream.eof() || result.empty()) {
        throw std::runtime_error("stage boundary table is malformed or empty");
    }
    return result;
}

std::vector<ObjectRow> read_objects(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open object table");
    std::vector<ObjectRow> objects;
    std::uint64_t index = 0;
    ObjectRow row;
    while (stream >> index >> row.base >> row.extent) {
        if (index != objects.size() || row.extent == 0 ||
            row.base > std::numeric_limits<std::uint64_t>::max() - row.extent) {
            throw std::runtime_error("object table is not dense or has an invalid extent");
        }
        objects.push_back(row);
    }
    if (!stream.eof() || objects.empty()) {
        throw std::runtime_error("object table is malformed or empty");
    }
    return objects;
}

std::vector<std::int64_t> read_object_period_deltas(
    const std::string& path, std::size_t object_count) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("cannot open affine-period delta table");
    std::vector<std::int64_t> deltas(object_count, 0);
    std::vector<bool> seen(object_count, false);
    std::uint64_t index = 0;
    std::string raw_delta;
    while (stream >> index >> raw_delta) {
        if (index >= object_count || seen[static_cast<std::size_t>(index)]) {
            throw std::runtime_error("affine-period delta table has a bad object index");
        }
        deltas[static_cast<std::size_t>(index)] =
            parse_i64(raw_delta, "affine-period object delta");
        seen[static_cast<std::size_t>(index)] = true;
    }
    if (!stream.eof() ||
        !std::all_of(seen.begin(), seen.end(), [](bool value) { return value; })) {
        throw std::runtime_error("affine-period delta table is malformed or incomplete");
    }
    return deltas;
}

void write_json_u64(std::ostream& out, const char* key, std::uint64_t value, bool& first) {
    if (!first) out << ',';
    first = false;
    out << '\n' << "    \"" << key << "\": " << value;
}

std::string json_escape(std::string_view value) {
    std::ostringstream result;
    for (const char item : value) {
        switch (item) {
            case '\\': result << "\\\\"; break;
            case '"': result << "\\\""; break;
            case '\n': result << "\\n"; break;
            case '\r': result << "\\r"; break;
            case '\t': result << "\\t"; break;
            default: result << item;
        }
    }
    return result.str();
}

void write_stage_counter_object(std::ostream& out, const Counters& c) {
    out << "{\"input_requests\":" << c.input_requests
        << ",\"input_bytes\":" << c.input_bytes
        << ",\"input_r_bytes\":" << c.input_r_bytes
        << ",\"input_w_bytes\":" << c.input_w_bytes
        << ",\"output_requests\":" << c.output_requests
        << ",\"output_bytes\":" << c.output_bytes
        << ",\"output_r_requests\":" << c.output_r_requests
        << ",\"output_r_bytes\":" << c.output_r_bytes
        << ",\"output_w_requests\":" << c.output_w_requests
        << ",\"output_w_bytes\":" << c.output_w_bytes
        << ",\"reason_read-miss_requests\":" << c.reason_read_miss
        << ",\"reason_dirty-eviction_requests\":" << c.reason_dirty_eviction
        << ",\"reason_write-allocate-fill_requests\":"
        << c.reason_write_allocate_fill
        << ",\"reason_write-through_requests\":" << c.reason_write_through
        << ",\"reason_write-around_requests\":" << c.reason_write_around
        << ",\"reason_final-dirty-drain_requests\":"
        << c.reason_final_dirty_drain
        << ",\"read_hits\":" << c.read_hits
        << ",\"read_misses\":" << c.read_misses
        << ",\"write_hits\":" << c.write_hits
        << ",\"write_misses\":" << c.write_misses
        << ",\"line_evictions\":" << c.line_evictions
        << ",\"dirty_sector_writebacks\":" << c.dirty_sector_writebacks
        << ",\"final_dirty_sector_writebacks\":"
        << c.final_dirty_sector_writebacks
        << ",\"sass_evict_first_read_accesses\":"
        << c.sass_evict_first_read_accesses
        << ",\"write_miss_allocations_without_fetch\":"
        << c.write_miss_allocations_without_fetch << "}";
}

void write_stage_rows(std::ostream& out, const std::vector<StageRow>& rows) {
    out << "  \"stage_audit\": {\n"
        << "    \"state_order\": \"per-set MRU-to-LRU\",\n"
        << "    \"stages\": [";
    for (std::size_t index = 0; index < rows.size(); ++index) {
        if (index != 0) out << ',';
        const auto& row = rows[index];
        out << "\n      {\"label\":\"" << json_escape(row.label)
            << "\",\"begin_input_records\":" << row.begin_input_records
            << ",\"end_input_records\":" << row.end_input_records
            << ",\"delta\":";
        write_stage_counter_object(out, row.delta);
        out << ",\"output_sha256\":\"" << row.output_sha256
            << "\",\"output_digest_records\":" << row.output_digest_records
            << ",\"state\":{\"resident_lines\":" << row.resident_lines
            << ",\"resident_sector_bytes\":" << row.resident_sector_bytes
            << ",\"resident_dirty_sector_bytes\":"
            << row.resident_dirty_sector_bytes
            << ",\"physical_mru_to_lru_sha256\":\""
            << row.physical_mru_to_lru_sha256
            << "\",\"semantic_contents_sha256\":\""
            << row.semantic_contents_sha256 << "\"}}";
    }
    if (!rows.empty()) out << '\n';
    out << "    ]\n  },\n";
}

void write_stats(
    const std::string& path,
    const Counters& c,
    const Cache& cache,
    const Output& output,
    std::uint64_t sector_bytes,
    const std::string& input_sha,
    const std::string& output_sha,
    const std::string& input_untimed_sha,
    const std::string& output_untimed_sha,
    bool timed_input_output,
    const AffinePeriodAudit* affine_period_audit,
    const std::vector<StageRow>* stage_rows) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write stats JSON");
    out << "{\n  \"schema\": {\"name\": \"hbfsim.fast_compact_cache_engine\", \"version\": 1},\n";
    out << "  \"status\": \"PASS\",\n";
    cache.write_read_fill_json(out);
    out << "  \"wire_record_bytes\": "
        << (timed_input_output ? sizeof(TimedRecord) : sizeof(Record)) << ",\n";
    out << "  \"timed_input_output\": "
        << (timed_input_output ? "true" : "false") << ",\n";
    out << "  \"input_sha256\": \"" << input_sha << "\",\n";
    out << "  \"output_sha256\": \"" << output_sha << "\",\n";
    out << "  \"input_untimed_sha256\": \"" << input_untimed_sha
        << "\",\n";
    out << "  \"output_untimed_sha256\": \"" << output_untimed_sha
        << "\",\n";
    out << "  \"counts\": {";
    bool first = true;
    write_json_u64(out, "input_requests", c.input_requests, first);
    write_json_u64(out, "input_bytes", c.input_bytes, first);
    write_json_u64(out, "input_r_bytes", c.input_r_bytes, first);
    write_json_u64(out, "input_w_bytes", c.input_w_bytes, first);
    write_json_u64(out, "output_requests", c.output_requests, first);
    write_json_u64(out, "output_bytes", c.output_bytes, first);
    write_json_u64(out, "output_r_requests", c.output_r_requests, first);
    write_json_u64(out, "output_r_bytes", c.output_r_bytes, first);
    write_json_u64(out, "output_w_requests", c.output_w_requests, first);
    write_json_u64(out, "output_w_bytes", c.output_w_bytes, first);
    write_json_u64(out, "reason_read-miss_requests", c.reason_read_miss, first);
    write_json_u64(out, "reason_dirty-eviction_requests", c.reason_dirty_eviction, first);
    write_json_u64(out, "reason_write-allocate-fill_requests", c.reason_write_allocate_fill, first);
    write_json_u64(out, "reason_write-through_requests", c.reason_write_through, first);
    write_json_u64(out, "reason_write-around_requests", c.reason_write_around, first);
    write_json_u64(out, "reason_final-dirty-drain_requests", c.reason_final_dirty_drain, first);
    out << "\n  },\n  \"cache_stats\": {";
    first = true;
    write_json_u64(out, "read_hits", c.read_hits, first);
    write_json_u64(out, "read_misses", c.read_misses, first);
    write_json_u64(out, "write_hits", c.write_hits, first);
    write_json_u64(out, "write_misses", c.write_misses, first);
    write_json_u64(out, "line_evictions", c.line_evictions, first);
    write_json_u64(out, "dirty_sector_writebacks", c.dirty_sector_writebacks, first);
    write_json_u64(out, "final_dirty_sector_writebacks", c.final_dirty_sector_writebacks, first);
    write_json_u64(out, "sass_evict_first_read_accesses", c.sass_evict_first_read_accesses, first);
    write_json_u64(out, "write_miss_allocations_without_fetch", c.write_miss_allocations_without_fetch, first);
    out << "\n  },\n  \"output_run_census\": {\n";
    out << "    \"records\": " << output.run_records() << ",\n";
    out << "    \"runs\": " << output.run_count() << ",\n";
    out << "    \"maximum_records_per_run\": "
        << output.maximum_run_length() << ",\n";
    out << "    \"length_histogram\": {";
    first = true;
    for (const auto& [length, count] : output.run_histogram()) {
        if (!first) out << ',';
        first = false;
        out << "\n      \"" << length << "\": " << count;
    }
    if (!first) out << '\n';
    out << "    }\n  },\n";
    if (affine_period_audit != nullptr) {
        out << "  \"affine_period_audit\": {\n";
        out << "    \"prefix_input_records\": "
            << affine_period_audit->prefix_input_records() << ",\n";
        out << "    \"region_input_records\": "
            << affine_period_audit->region_input_records() << ",\n";
        out << "    \"region_output_requests\": "
            << affine_period_audit->region_output_requests() << ",\n";
        out << "    \"region_read_requests\": "
            << affine_period_audit->region_read_requests() << ",\n";
        out << "    \"region_write_requests\": "
            << affine_period_audit->region_write_requests() << ",\n";
        out << "    \"affine_region_output_sha256\": \""
            << affine_period_audit->region_output_sha256() << "\",\n";
        out << "    \"input_records_per_period\": "
            << affine_period_audit->input_records_per_period() << ",\n";
        out << "    \"object_period_delta_bytes\": {";
        first = true;
        const auto& deltas = affine_period_audit->object_period_deltas();
        for (std::size_t index = 0; index < deltas.size(); ++index) {
            if (!first) out << ',';
            first = false;
            out << "\n      \"" << index << "\": " << deltas[index];
        }
        if (!first) out << '\n';
        out << "    },\n    \"periods\": [";
        first = true;
        for (const auto& row : affine_period_audit->rows()) {
            if (!first) out << ',';
            first = false;
            out << "\n      {\"period_index\": " << row.period_index
                << ", \"input_requests\": " << row.input_requests
                << ", \"output_requests\": " << row.output_requests
                << ", \"read_requests\": " << row.read_requests
                << ", \"write_requests\": " << row.write_requests
                << ", \"normalized_read_sha256\": \""
                << row.normalized_read_sha256 << "\"}";
        }
        if (!first) out << '\n';
        out << "    ]";
        if (affine_period_audit->writes_artifacts()) {
            out << ",\n    \"macro_artifacts\": {\n";
            out << "      \"read_template\": {\"path\": \""
                << affine_period_audit->read_template_path()
                << "\", \"records\": "
                << affine_period_audit->read_template_records()
                << ", \"record_bytes\": " << sizeof(Record)
                << ", \"sha256\": \""
                << affine_period_audit->read_template_sha256() << "\"},\n";
            out << "      \"writeback_fallback\": {\"path\": \""
                << affine_period_audit->writeback_fallback_path()
                << "\", \"records\": "
                << affine_period_audit->writeback_fallback_records()
                << ", \"record_bytes\": " << sizeof(AffineFallbackRecord)
                << ", \"sha256\": \""
                << affine_period_audit->writeback_fallback_sha256() << "\"}\n";
            out << "    }";
        }
        out << "\n  },\n";
    }
    if (stage_rows != nullptr) {
        write_stage_rows(out, *stage_rows);
    }
    out << "  \"final_state\": {\n";
    out << "    \"resident_lines\": " << cache.resident_lines() << ",\n";
    out << "    \"resident_sector_bytes\": " << cache.resident_sectors() * sector_bytes << ",\n";
    out << "    \"resident_dirty_sector_bytes\": "
        << cache.resident_dirty_sectors() * sector_bytes;
    if (stage_rows != nullptr && !stage_rows->empty()) {
        const auto& final = stage_rows->back();
        out << ",\n    \"physical_mru_to_lru_sha256\": \""
            << final.physical_mru_to_lru_sha256
            << "\",\n    \"semantic_contents_sha256\": \""
            << final.semantic_contents_sha256 << "\"";
    }
    out << "\n";
    out << "  }\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        std::string objects_path;
        std::string stats_path;
        std::string affine_period_deltas_path;
        std::string affine_read_template_output;
        std::string affine_writeback_fallback_output;
        std::string stage_boundaries_path;
        std::uint64_t capacity = 0;
        std::uint64_t line = 0;
        std::uint64_t sector = 0;
        std::uint64_t read_fill_bytes = 0;
        std::uint64_t writeback_bytes = 0;
        bool clip_read_fill_to_object = false;
        std::uint64_t associativity = 0;
        std::uint64_t affine_prefix_input_records = 0;
        std::uint64_t affine_period_input_records = 0;
        bool write_back = true;
        bool write_allocate = true;
        bool write_miss_fetch = true;
        bool final_drain = false;
        bool timed_input_output = false;
        for (int index = 1; index < argc; ++index) {
            const std::string key = argv[index];
            if (index + 1 >= argc) throw std::runtime_error("missing command-line value");
            const std::string value = argv[++index];
            if (key == "--objects") objects_path = value;
            else if (key == "--stats") stats_path = value;
            else if (key == "--capacity-bytes") capacity = parse_u64(value, "capacity");
            else if (key == "--line-bytes") line = parse_u64(value, "line bytes");
            else if (key == "--sector-bytes") sector = parse_u64(value, "sector bytes");
            else if (key == "--read-fill-bytes") read_fill_bytes = parse_u64(value, "read fill bytes");
            else if (key == "--writeback-bytes") writeback_bytes = parse_u64(value, "writeback bytes");
            else if (key == "--read-fill-object-edge") {
                if (value == "clip") clip_read_fill_to_object = true;
                else if (value == "reject") clip_read_fill_to_object = false;
                else throw std::runtime_error("read-fill object edge must be reject or clip");
            }
            else if (key == "--associativity") associativity = parse_u64(value, "associativity");
            else if (key == "--affine-period-input-records") {
                affine_period_input_records =
                    parse_u64(value, "affine-period input records");
            } else if (key == "--affine-prefix-input-records") {
                affine_prefix_input_records =
                    parse_u64(value, "affine-prefix input records");
            } else if (key == "--affine-period-object-deltas") {
                affine_period_deltas_path = value;
            } else if (key == "--affine-read-template-output") {
                affine_read_template_output = value;
            } else if (key == "--affine-writeback-fallback-output") {
                affine_writeback_fallback_output = value;
            } else if (key == "--stage-boundaries") {
                stage_boundaries_path = value;
            }
            else if (key == "--write-policy") {
                if (value == "write-back") write_back = true;
                else if (value == "write-through") write_back = false;
                else throw std::runtime_error("invalid write policy");
            } else if (key == "--write-allocate") write_allocate = parse_bool(value, "write allocate");
            else if (key == "--write-miss-fetch") write_miss_fetch = parse_bool(value, "write miss fetch");
            else if (key == "--final-drain") final_drain = parse_bool(value, "final drain");
            else if (key == "--timed-input-output") {
                timed_input_output = parse_bool(value, "timed input/output");
            }
            else throw std::runtime_error("unknown command-line option: " + key);
        }
        if (objects_path.empty() || stats_path.empty()) {
            throw std::runtime_error("--objects and --stats are required");
        }
        if ((affine_period_input_records == 0) != affine_period_deltas_path.empty()) {
            throw std::runtime_error(
                "affine-period input records and object deltas must be supplied together");
        }
        if (affine_prefix_input_records != 0 && affine_period_input_records == 0) {
            throw std::runtime_error(
                "affine-prefix input records require the affine-period audit");
        }
        if (affine_read_template_output.empty() !=
            affine_writeback_fallback_output.empty()) {
            throw std::runtime_error(
                "affine read-template and writeback-fallback outputs must be paired");
        }
        if (!affine_read_template_output.empty() &&
            affine_period_input_records == 0) {
            throw std::runtime_error(
                "affine macro outputs require the affine-period audit");
        }
        if (affine_period_input_records != 0 && final_drain) {
            throw std::runtime_error(
                "affine-period audit does not permit an unassigned final cache drain");
        }
        if (!stage_boundaries_path.empty() && final_drain) {
            throw std::runtime_error(
                "stage-boundary audit does not permit an unassigned final cache drain");
        }
        if (!stage_boundaries_path.empty() && affine_period_input_records != 0) {
            throw std::runtime_error(
                "stage-boundary and affine-period audits cannot share one run");
        }
        if (timed_input_output && affine_period_input_records != 0) {
            throw std::runtime_error(
                "timed input/output cannot be combined with affine-period artifacts");
        }
        if (timed_input_output && final_drain) {
            throw std::runtime_error(
                "timed input/output requires final-drain=false");
        }

        Counters counters;
        Sha256 input_digest;
        Sha256 output_digest;
        Sha256 input_untimed_digest;
        Sha256 output_untimed_digest;
        auto objects = read_objects(objects_path);
        const auto stage_boundaries = read_stage_boundaries(stage_boundaries_path);
        std::unique_ptr<StageOutputTracker> stage_output_tracker;
        if (!stage_boundaries.empty()) {
            stage_output_tracker = std::make_unique<StageOutputTracker>();
        }
        std::unique_ptr<AffinePeriodAudit> affine_period_audit;
        if (affine_period_input_records != 0) {
            affine_period_audit = std::make_unique<AffinePeriodAudit>(
                affine_prefix_input_records,
                affine_period_input_records,
                read_object_period_deltas(
                    affine_period_deltas_path, objects.size()),
                affine_read_template_output,
                affine_writeback_fallback_output);
        }
        Output output(
            stdout, counters, output_digest, output_untimed_digest,
            timed_input_output, affine_period_audit.get(),
            stage_output_tracker.get());
        Cache cache(
            std::move(objects), capacity, line, sector, associativity,
            write_back, write_allocate, write_miss_fetch, counters, output,
            read_fill_bytes, clip_read_fill_to_object, writeback_bytes);
        std::vector<StageRow> stage_rows;
        std::size_t next_stage = 0;
        Counters stage_begin_counters;
        std::uint64_t stage_begin_input_records = 0;

        auto process_record = [&](const Record& record, std::uint64_t issue_ns) {
                    input_untimed_digest.update(&record, sizeof(record));
                    cache.access(record, issue_ns);
                    if (affine_period_audit != nullptr) {
                        affine_period_audit->finish_input_record(
                            counters.input_requests);
                    }
                    if (next_stage < stage_boundaries.size()) {
                        const auto& boundary = stage_boundaries[next_stage];
                        if (counters.input_requests > boundary.end_input_records) {
                            throw std::runtime_error(
                                "input crossed a stage boundary without an exact match");
                        }
                        if (counters.input_requests == boundary.end_input_records) {
                            const auto [stage_output_sha, stage_output_records] =
                                stage_output_tracker->finish_stage();
                            const auto delta = subtract_counters(
                                counters, stage_begin_counters);
                            if (stage_output_records != delta.output_requests) {
                                throw std::runtime_error(
                                    "stage output digest does not conserve output records");
                            }
                            stage_rows.push_back(StageRow{
                                .label = boundary.label,
                                .begin_input_records = stage_begin_input_records,
                                .end_input_records = boundary.end_input_records,
                                .delta = delta,
                                .output_sha256 = stage_output_sha,
                                .output_digest_records = stage_output_records,
                                .resident_lines = cache.resident_lines(),
                                .resident_sector_bytes =
                                    cache.resident_sectors() * sector,
                                .resident_dirty_sector_bytes =
                                    cache.resident_dirty_sectors() * sector,
                                .physical_mru_to_lru_sha256 =
                                    cache.physical_mru_to_lru_sha256(),
                                .semantic_contents_sha256 =
                                    cache.semantic_contents_sha256(),
                            });
                            stage_begin_counters = counters;
                            stage_begin_input_records = boundary.end_input_records;
                            ++next_stage;
                        }
                    }
        };

        constexpr std::size_t kChunkRecords = 1U << 20;
        if (timed_input_output) {
            std::vector<TimedRecord> records(kChunkRecords);
            while (true) {
                const auto byte_count = std::fread(
                    records.data(), 1,
                    records.size() * sizeof(TimedRecord), stdin);
                if (byte_count % sizeof(TimedRecord) != 0) {
                    throw std::runtime_error("truncated timed compact input record");
                }
                const auto count = byte_count / sizeof(TimedRecord);
                if (count != 0) {
                    input_digest.update(
                        records.data(), count * sizeof(TimedRecord));
                    for (std::size_t index = 0; index < count; ++index) {
                        process_record(records[index].request, records[index].issue_ns);
                    }
                }
                if (byte_count < records.size() * sizeof(TimedRecord)) {
                    if (std::ferror(stdin)) {
                        throw std::runtime_error("failed to read timed compact input");
                    }
                    if (!std::feof(stdin)) {
                        throw std::runtime_error(
                            "short timed compact read without EOF");
                    }
                    break;
                }
            }
        } else {
            std::vector<Record> records(kChunkRecords);
            while (true) {
                const auto byte_count = std::fread(
                    records.data(), 1,
                    records.size() * sizeof(Record), stdin);
                if (byte_count % sizeof(Record) != 0) {
                    throw std::runtime_error("truncated compact input record");
                }
                const auto count = byte_count / sizeof(Record);
                if (count != 0) {
                    input_digest.update(records.data(), count * sizeof(Record));
                    for (std::size_t index = 0; index < count; ++index) {
                        process_record(records[index], 0);
                    }
                }
                if (byte_count < records.size() * sizeof(Record)) {
                    if (std::ferror(stdin)) {
                        throw std::runtime_error("failed to read compact input");
                    }
                    if (!std::feof(stdin)) {
                        throw std::runtime_error(
                            "short compact read without EOF");
                    }
                    break;
                }
            }
        }
        if (final_drain) cache.drain();
        if (!stage_boundaries.empty()) {
            if (next_stage != stage_boundaries.size() ||
                stage_boundaries.back().end_input_records != counters.input_requests ||
                stage_output_tracker->records() != 0) {
                throw std::runtime_error(
                    "stage boundaries do not end exactly at the input stream EOF");
            }
        }
        if (affine_period_audit != nullptr) {
            affine_period_audit->finish(counters.input_requests);
        }
        output.finish_runs();
        output.flush();
        if (std::fflush(stdout) != 0) throw std::runtime_error("failed to flush compact output");
        write_stats(
            stats_path, counters, cache, output, sector,
            input_digest.finish(), output_digest.finish(),
            input_untimed_digest.finish(), output_untimed_digest.finish(),
            timed_input_output,
            affine_period_audit.get(),
            stage_boundaries.empty() ? nullptr : &stage_rows);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "fast_cache_compact_trace: " << error.what() << '\n';
        return 1;
    }
}
