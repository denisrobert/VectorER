# Security Policy

## Supported Versions

`vector-er` is **beta software** (pre-`1.0.0`). Until the `1.0.0` release:

- **Only the latest released version is supported.** Security fixes are
  released only against the current version and are **not** backported to
  earlier releases.
- Past versions are not supported. Dependents should upgrade to the latest
  release to receive fixes.

| Version                | Supported          |
|------------------------|--------------------|
| latest release (0.x)   | Supported          |
| earlier releases (0.x) | Not supported      |

## Reporting a Vulnerability

Please report suspected security vulnerabilities **privately**, using
[GitHub private vulnerability reporting][gh-private-reporting] on this
repository:

1. Go to the repository's **Security** tab
   (<https://github.com/denisrobert/VectorER/security/advisories>).
2. Click **Report a vulnerability** and fill in the details.

Do **not** open a public issue for a suspected security vulnerability; private
reporting gives maintainers a chance to address it before details are public.

### What to include

- The affected `vectorer` version and your environment (OS, Python version,
  dependency versions — see the bug report template for the full list).
- A minimal, self-contained description of the vulnerability and, where
  possible, a proof of concept (reproduction script or steps).
- Any impact assessment you can provide (attack surface, data exposure,
  severity).

### Response expectations

Until the `1.0.0` release, **no response-time guarantee is made**: reports are
acknowledged and triaged as soon as maintainer time allows. After triage you
will receive updates on the fix and, when a fix is released, coordination on
public disclosure.

[gh-private-reporting]: https://docs.github.com/en/code-security/security-advisories/working-with-repository-security-advisories/configuring-private-vulnerability-reporting-for-a-repository