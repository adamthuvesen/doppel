Output assets/demo.gif

Set Shell "zsh"
Set FontSize 13
Set Width 920
Set Height 650
Set Theme "AtomOneLight"
Set Padding 20
Set TypingSpeed 35ms
Set PlaybackSpeed 1

Hide
Type "cd /path/to/doppel && export PS1='$ ' && clear"
Enter
Sleep 500ms
Show

Sleep 400ms

Type "uv run doppel gen examples/saas_accounts.csv -n 1000 -o /tmp/synth.csv --seed 1"
Sleep 600ms
Enter
Sleep 9s

Type "uv run doppel diff examples/saas_accounts.csv /tmp/synth.csv --max-marginal 0.10 --min-dcr-p5 0.03"
Sleep 600ms
Enter
Sleep 10s

Sleep 6s
