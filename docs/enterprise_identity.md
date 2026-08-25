# Enterprise identity and control-plane architecture

This document is the decision authority for the first E2 implementation. It
defines trust boundaries; it is not a deployment tutorial and contains no
credential, customer identifier, connection string, or content field.

## Pilot identity provider and browser session

The pilot provider is **Keycloak**, used through standards-based OpenID
Connect discovery. Deployments configure one exact HTTPS issuer and one exact
client identifier. The backend-for-frontend (BFF), not browser JavaScript,
exchanges an authorization code and holds provider tokens server-side.

The browser receives only an opaque session cookie. It is `Secure`,
`HttpOnly`, and `SameSite=Lax`; state-changing requests also require a
session-bound CSRF value. Login uses authorization code flow with PKCE `S256`.
The callback accepts a code only with the same one-use state, nonce, redirect
URI, and verifier issued for that browser attempt. State and code redemption
are single-use.

The verifier accepts only `RS256` identity tokens whose signature matches a
key from the configured issuer's discovered `jwks_uri`. It checks exact
issuer, client/audience, expiry, issued-at, nonce, and subject. Unknown keys
trigger one bounded JWKS refresh; keys disappear after the configured overlap
window and are never accepted from an untrusted token header URL. Discovery,
JWKS, token, and authorization endpoints must remain under the configured
issuer policy. Algorithm substitution, unsigned tokens, request-selected
tenants and token-provided product roles fail closed.

Keycloak is the first interoperable target, not a code-level default. The
same closed verifier contract can support another approved OIDC provider
without changing the product principal or authorization rules.

## One product principal

OIDC users and the retained signed OpenWebUI bridge both resolve to the same
closed product principal. An external `(issuer, subject)` identifies a human
only; tenant membership, product role, position, protected status and
architect capability are database authority. Display name, email, IdP group,
OpenWebUI role and request headers never grant product authority.

Service accounts are a distinct principal kind. A legacy static API key is
not a service account. Issuance, rotation and revocation use stored digests,
closed scopes, expiry and immutable audit evidence; a revoked or superseded
credential fails before data-plane checkout.

## Content-free control plane

E2 introduces a separate control-plane repository and connection pool. Its
runtime connection comes from `PG_CONTROL_DSN`; migrations use
`PG_CONTROL_MIGRATION_DSN`. Enterprise deployments keep this database
physically separate from tenant data planes. Local and team profiles may
co-locate it behind the same repository interface, but there is no fallback
from a missing control-plane route to `PG_DSN`.

The control plane may contain only opaque operational facts:

- tenant id, lifecycle state and deployment profile;
- region code, route kind, revision and opaque connection reference;
- feature assignments and closed quota values;
- HMAC identity-routing digests and platform-operator role;
- policy/configuration revisions and immutable administrative events.

It never stores prompts, messages, documents, chunks, embeddings, filenames,
email addresses, display names, raw issuer/subject values, secrets, DSNs,
bucket names, object keys or arbitrary JSON. A connection reference is an
opaque lookup key for the deployment secret system, never a credential.

Platform administration and tenant administration are different
capabilities. Tenant `admin`, tenant root and organization architect do not
imply platform authority. Platform operators can manage content-free routing
and lifecycle facts but cannot read customer content.

Quota records are descriptive in E2 and report `quota_enforcement=declared`.
Enforcement and fair scheduling belong to E5 and must not be implied early.

## Break-glass boundary

Break-glass is never a standing role. A platform-authorized operator requests
one tenant, one closed purpose, one reason, one short expiry and one incident
reference. Approval, use and expiry create immutable content-free events.
The capability cannot be renewed implicitly, cannot widen tenant scope, and
does not bypass protected-manager or content audit rules. Normal principals
remain unchanged when the grant expires or is revoked.

## Deployment contract

The repository exposes no usable OIDC or control-plane default. Operators
must supply the exact values in `.env.example`. HTTPS is required outside a
literal loopback development environment. Browser and provider URLs are
validated as origins/absolute endpoints rather than joined from untrusted
request data.

References for the selected pilot provider:

- Keycloak OIDC endpoints and discovery:
  https://www.keycloak.org/securing-apps/oidc-layers
- Keycloak client PKCE setting (`S256`) and realm JWKS endpoint:
  https://www.keycloak.org/docs/latest/server_admin/
