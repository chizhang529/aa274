### Submission Instructions

#### Writeup
All solution writeups (math, plots, explanations) in this class must be submitted electronically as a .pdf file. Preferably this writeup will be produced using LaTeX or Word, but if your submission will be handwritten and scanned, please use a flatbed scanner or a scanning app on your phone (raw photos will not be accepted). Name your writeup `<your-sunet-id>.pdf` (e.g., `pavone.pdf`) and place it in the base level of this directory (i.e., at `.../AA274_HW1/pavone.pdf`).

#### ROS code/bags
The submission script will search your catkin_workspace for the first ROS bag files it encounters that contain the random_strings topic and the turtlebot state/control topics. If you have multiple bag files and would like to choose which ones you submit, you may place your bags at `~/catkin_ws/random_strings.bag` and `~/catkin_ws/turtlebot.bag`.

#### Submission
To download the submission script, open this directory (i.e., `.../AA274_HW1/`) in your terminal and run `git pull` at the command line. Then run `./submit_hw1_code_and_writeup.sh` and follow the instructions to submit!