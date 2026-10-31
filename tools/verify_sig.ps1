# Usage: powershell -NoProfile -ExecutionPolicy Bypass -File verify_sig.ps1 <file>
# Prints the Authenticode status and signer; exits 1 unless the signature is valid and Microsoft's.
param([string]$Path)
$s = Get-AuthenticodeSignature -LiteralPath $Path
"file: $Path"
"size: $((Get-Item -LiteralPath $Path).Length)"
"signature: $($s.Status)"
"signer: $($s.SignerCertificate.Subject)"
if ($s.Status -ne 'Valid' -or $s.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') { exit 1 }
exit 0
