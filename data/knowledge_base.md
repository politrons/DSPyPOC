# SaaS E-Invoicing Operations Handbook

## Stamping Lifecycle

Invoice stamping is asynchronous. The normal states are `Draft -> Queued -> Processing -> Issued`.
If an invoice remains in `Processing` for more than 10 minutes, support should inspect provider latency and retry logs.
If `Processing` exceeds 30 minutes, open an incident with provider ticket and document IDs.

## Certificate Rotation

Certificates must be rotated at least 7 days before expiration. The platform requires:
- New certificate file
- Private key
- Password validation check
After rotation, run a test issuance in sandbox and production.

## Webhook Delivery

Webhook delivery is at-least-once. Consumer services must implement idempotency using `event_id`.
A delivery is considered healthy when endpoint p95 latency is below 3 seconds and HTTP success rate is above 99%.
For rotating webhook secrets, allow dual-secret validation during a transition window.

## API Authorization

Invoice creation requires scope `write:invoices` and tenant header `X-Tenant-Id`.
Status query by UUID requires scope `read:invoices`.
A 401 usually indicates invalid credentials. A 403 usually indicates insufficient scope or role.

## Numbering and Compliance

Numbering sequences can be configured per business unit.
Sequence gaps must be audited and justified to satisfy external compliance checks.
Invoice cancellation must preserve traceability to original UUID and cancellation reason.

## Performance at Month-End

At month-end, bulk issuance should use asynchronous queue processing.
Recommended controls:
- Per-tenant concurrency cap.
- Exponential backoff retries with jitter.
- Queue depth and provider latency monitoring.

## Email Delivery of Invoices

When customers do not receive invoice emails, verify:
- SMTP response code and bounce type.
- SPF, DKIM, and DMARC alignment.
- Recipient domain-level filters.
After fixing delivery settings, resend only failed documents to avoid duplicates.

## Audit Exports

For external audits, export:
- Signed CSV with invoice states.
- Change log with actor and timestamp.
- Reconciliation report by issue date and payment date.
Retention period for audit logs should be at least 5 years unless local regulation requires more.
