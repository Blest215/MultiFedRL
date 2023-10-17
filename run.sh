#!/bin/bash
  
rm log.txt

kill $(ps aux | grep '[p]ython main.py' | awk '{print $2}')

nohup python main.py >> log.txt &
