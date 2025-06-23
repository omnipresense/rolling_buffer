# rolling_buffer
Python code to set OPS243-A sensor into Rolling Buffer mode for high speed data capture.  High speed events include capturing muzzle velocity of bullets, baseball/bat, golf balls, or hocky pucks.  The sensor continuously samples until a trigger signal is provided upon which data is captured and output for post processing.  Python code will do post processing, running the captured data thru an FFT to report the speed values capatured which are provided along with a plot.

Type "python rolling_buffer.py" in terminal window to start the Python code processing.  Enter 'Trig' and enter to capture data.
