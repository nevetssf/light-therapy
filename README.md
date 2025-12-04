# light-therapy

Characterization of Light Therapy Tools

## Scope of project

To characterize the power and spectrum output of various light therapy devices for consumer use. 

Most consumer-grade light therapy devices are vague about the exact spectrum of their devices, and often don't mention the power density at all, both of which are critical for the intended therapy. 

I'll start by building a notebook which analyzes the `CSV` files created by the HP330. They're in a slightly odd format, so I need to do a bit of parsing to get them in a clean tabular format. 

Because of the intensities involved, I'll also use a few different techniques to attenuate the measurement. The simplest is just a piece of paper, but I'll need to calibrate the attenuation factor. 


## HP330 Photospectrometer

Bought on AliExpress, seems to be well-calibrated. It measures power density from 380 to 780 nm in default units of mW/m^2/nm. 

