param(
    [string]$InputPath = ".\trainlog_202609011654.txt",
    [string]$OutputPath = ".\trainlog_202609011654_sanitized.csv"
)

$lines = Get-Content -LiteralPath $InputPath -Encoding UTF8
$rows = [System.Collections.Generic.List[object]]::new()
$round = $null
$redactionCount = 0

function Sanitize-Line([string]$Text) {
    $patterns = @(
        @{ Pattern = '(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'; Replacement = '<EMAIL>' },
        @{ Pattern = '(?<![\d.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![\d.])'; Replacement = '<IP>' },
        @{ Pattern = '(?i)\b(?:[0-9A-F]{2}[:-]){5}[0-9A-F]{2}\b'; Replacement = '<MAC>' },
        @{ Pattern = '(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b'; Replacement = '<UUID>' },
        @{ Pattern = '(?i)\bhttps?://[^\s,;]+'; Replacement = '<URL>' },
        @{ Pattern = '(?i)(?:[A-Z]:\\|/)(?:[^\s<>:"|?*]+[\\/])+[^\s<>:"|?*]*'; Replacement = '<PATH>' },
        @{ Pattern = '(?i)(\b(?:api[_-]?key|access[_-]?token|secret|password|passwd)\s*[=:]\s*)[^\s,;]+'; Replacement = '$1<SECRET>' },
        @{ Pattern = '(?i)(\b(?:user(?:name)?|account|host(?:name)?|device[_-]?id)\s*[=:]\s*)[^\s,;]+'; Replacement = '$1<IDENTIFIER>' }
    )
    $result = $Text
    foreach ($item in $patterns) {
        $before = $result
        $result = [regex]::Replace($result, $item.Pattern, $item.Replacement)
        if ($result -ne $before) { $script:redactionCount++ }
    }
    return $result
}

for ($i = 0; $i -lt $lines.Count; $i++) {
    $sanitized = Sanitize-Line $lines[$i]
    if ($sanitized -match '(?i)\bround\s*[=:]\s*(\d+)') { $round = [int]$Matches[1] }
    elseif ($sanitized -match '^t\s+(\d+)\s*:') { $round = [int]$Matches[1] }

    $category = if ($sanitized -match '^\[([^\]]+)\]') { $Matches[1] } elseif ($sanitized -match '^t\s+\d+') { 'training' } elseif ($sanitized -match '^(attack|defend)\s*$') { 'phase' } else { 'general' }
    $event = if ($sanitized -match '^\[[^\]]+\]\s*(.*)$') { $Matches[1] } else { $sanitized }
    $numberMatches = [regex]::Matches($sanitized, '(?i)(?<![A-Za-z_])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?|(?<![A-Za-z_])[-+]?(?:nan|inf)(?![A-Za-z_])')

    if ($numberMatches.Count -eq 0) {
        $rows.Add([pscustomobject]@{ line_no = $i + 1; round = $round; category = $category; event = $event; numeric_index = $null; numeric_value = $null; sanitized_text = $sanitized })
        continue
    }

    for ($j = 0; $j -lt $numberMatches.Count; $j++) {
        $value = $numberMatches[$j].Value -replace ',', ''
        $rows.Add([pscustomobject]@{ line_no = $i + 1; round = $round; category = $category; event = $event; numeric_index = $j + 1; numeric_value = $value; sanitized_text = $sanitized })
    }
}

$csv = $rows | ConvertTo-Csv -NoTypeInformation
[System.IO.File]::WriteAllLines((Join-Path (Get-Location) $OutputPath), $csv, [System.Text.UTF8Encoding]::new($true))

[pscustomobject]@{
    source_lines = $lines.Count
    csv_rows = $rows.Count
    redaction_rules_triggered = $redactionCount
    output = (Resolve-Path -LiteralPath $OutputPath).Path
}
