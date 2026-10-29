# Errors and exit codes

Every failure prints one `error:` line and, where possible, a `hint:` line. Tracebacks appear only with `--debug`.

| Code | Meaning |
|---|---|
| 0 | success |
| 2 | usage error (bad flag, unknown command) |
| 3 | input error (missing file, unreadable format, empty corpus) |
| 4 | model or optional dependency missing, or offline |
| 5 | resources (out of memory, disk) |
| 70 | unexpected failure (a bug, please report it with `--debug` output) |
| 130 | interrupted |
