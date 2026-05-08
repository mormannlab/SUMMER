# Experiment Code

Executable files and base code for programs used in the movie presentation paradigm. 

The base code is compressed to decrease the overall size of the code repository. 

---


## Folder structure

```
paradigm_code/
├── modified_ffmpeg.zip         # compressed base code for the modified FFPLAY version
└── timedDAQ                    # clocking application, compiled executable file 

```
---

This directory includes the two pieces of software used to present the movie and track stimulus event timestamps alongside the neural activity. 
Both programs were run on a laptop while connected to the neural recording system.

### timedDAQ

This program sends a numbered event every second to the neural recording system (ATLAS, Neuralynx). The program logs the laptop's own timestamp each time an event is transmitted to the neural recording system and writes these timestamps to the outputted log, `timedDAQ-log-<YYYYMMDD-HHMMSS>.log`. The transmitted event timestamps are collected by the neural recording system and saved to a file called `Events.nev`. These logs are co-registered, allowing synchronization between the movie content and the neural activity. 

This program was written by Johannes Niediek, and last updated in 2019. The source code is available [here](https://github.com/jniediek/DAQ1208FS/tree/master/timedDAQ). 

### modified FFMPEG

This program is a modified version of the open-source software [FFmpeg](https://www.ffmpeg.org/). FFmpeg is a suite of audio and video presentation tools. 

We altered the source code for the `FFplay` component to produce a log file tracking the frame presentation times. Once compiled, the movie is launched from the command line (e.g. `ffplay 500DaysOfSummer.avi`), which initiates a log file, `ffplay-watchlog-<YYYYMMDD-HHMMSS>.log`. This logfile tracks the presentation time stamps (PTS, time relative to the onset of the movie) and the local laptop time when the PTS was shown. It also tracks any interaction events, such as full-screening, pausing/playing, and rewinding/fast-forwarding. Note that the PTS indicates the frame id (frame index = PTS / frame rate). 

This program was originally modified to produce the watchlog by Johannes Niediek in 2012. It was updated by Alana Darcher in 2020 to fix an issue with the rewinding/fast-forwarding. 