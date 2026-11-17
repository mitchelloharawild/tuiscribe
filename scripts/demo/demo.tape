# Recorded by make-demo.sh, which sets up the virtual devices, the audio and the
# DEMO_* variables used below. Run that rather than `vhs` directly.

Output assets/demo/tuiscribe.gif

Set Shell bash
Set FontSize 15
Set Width 1100
Set Height 760
Set Padding 16
Set Framerate 12
Set TypingSpeed 60ms
Set Theme "Catppuccin Mocha"

# Off camera: route `tuiscribe` to the virtual devices and the demo settings,
# and start the dialogue playing a few seconds from now.
Hide
Type `cd "$DEMO_SESSION" && tuiscribe() { "$DEMO_TUISCRIBE" "$@" --mic tuiscribe-demo-mic --system tuiscribe-demo-sys --model base.en --gap-long 3; }`
Enter
Type `setsid -f "$DEMO_PLAY" "$DEMO_PLAY_DELAY" >/dev/null 2>&1; clear`
Enter
Sleep 500ms
Show

Type "tuiscribe record -o standup.md"
Sleep 400ms
Enter

# The dialogue: live partials settle into You / Remote lines, then a pause.
Sleep 44s

Type "n"
Sleep 600ms
Type "Decision: the Sun is stronger"
Sleep 500ms
Enter
Sleep 2.5s

Type "q"
Sleep 3.5s
