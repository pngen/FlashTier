# Security Policy

## Supported versions

Security fixes are applied to the latest release and to the `main` branch.

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes (latest release line) |

## Reporting a vulnerability

FlashTier is a local systems runtime. It does not collect, transmit, or
expose data to the network, and it has no server, daemon, or RPC surface.
Please still report any issue you find.

- Do **not** open a public issue for vulnerabilities involving remote
  execution, privilege escalation, or data exposure.
- Email the maintainer (Paul Ngen) through the GitHub profile at
  https://github.com/pngen/FlashTier (or use the private security advisory
  flow on GitHub: *Security → Report a vulnerability*).
- Include: affected version, operating system and GPU/driver when relevant,
  a minimal reproduction, and the impact you believe the issue has.

You will receive a response within 7 days; the disclosure timeline is
coordinated with you.

## Security considerations in this project

- Benchmarks allocate large amounts of VRAM, RAM, and disk. Sizes are
  validated against detected free capacity before allocation, and reserve
  margins are enforced — do not weaken these checks.
- NVMe backing stores are created in operator-chosen paths only; never
  write outside the configured store path.
- Telemetry is written only to locally chosen files; never add network
  telemetry, analytics, or external data collection.
- Do not commit secrets, keys, or machine-specific absolute paths.
