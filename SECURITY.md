# Security policy

HBServe parses model, request, router, placement, and simulator configuration
files as untrusted data, but it is not a security sandbox. Only run a simulator
binary you trust. Avoid publishing result bundles until their input artifact
paths and metadata have been reviewed for sensitive information.

Please report vulnerabilities through GitHub's private vulnerability reporting
for this repository. Do not include production traces, tokens, or credentials
in a public issue.
