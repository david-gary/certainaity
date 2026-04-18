# CLI Reference

## `forenscope analyze`

Run a full forensic analysis on a local image file.

```
forenscope analyze <image> [options]
```

### Arguments

| Argument | Description |
|----------|-------------|
| `image` | Path to the image file (JPEG, PNG, TIFF, WebP) |

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--job-id ID` | filename stem | Identifier used in the report and output filenames |
| `--output-json PATH` | — | Write the JSON report to `PATH` |
| `--output-pdf PATH` | — | Write the PDF report to `PATH` |

### Exit codes

| Code | Meaning |
|------|---------|
| `0` | Analysis completed (manipulation may or may not have been detected) |
| `1` | Error: file not found, corrupt image, or pipeline failure |

### Examples

```bash
# Print JSON to stdout
forenscope analyze photo.jpg

# Write both report formats
forenscope analyze photo.jpg \
  --job-id case-2026-001 \
  --output-json reports/case-2026-001.json \
  --output-pdf  reports/case-2026-001.pdf

# Use in a shell pipeline
forenscope analyze photo.jpg | jq '.manipulation_detected'
```

### Output schema

```json
{
  "job_id": "photo",
  "file_name": "photo.jpg",
  "sha256": "a3f4...",
  "analysis_timestamp": "2026-03-01T10:23:45.123456+00:00",
  "manipulation_detected": false,
  "overall_confidence": 0.12,
  "anti_forensic_warning": false,
  "regions": 0,
  "execution_time_ms": 843
}
```

!!! note
    `regions` is the count of detected manipulation regions, not the full region
    list. Use `--output-json` to get the complete region details including bounding
    boxes, types, and evidence strings.
