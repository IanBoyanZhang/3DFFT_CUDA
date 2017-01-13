#!/bin/csh
# This a batch submission script for a CUDA Job runing on Dirac
#  Submit this script using the command: qsub <script_name>
#  Use the "qstat" command to check the status of a job.
#
# There are only a few places you'll need to make changes, as marked below
 

# 1. The wall clock job time limit
# 2. The job name
# 3. An email notification when the job is complete
#   !!! Change to the address you want the notification sent to !!!
# 4. The job


#
# The following are embedded QSUB options. The syntax is #PBS (the # does
# _not_  denote that the lines are commented out so do not remove).
#

# Don't change this part of the batch submission file
#

#
# The job queue
#PBS -q dirac_reg
#
# Ask for 1 fermi node, all 8 CPU cores
#PBS -l nodes=1:ppn=8:fermi




# Job output files: you get separate files for stdout and stderr
#
# Filename for standard output (default = <job_name>.o<job_id>)
# at end of job, it is in directory from which qsub was executed

# Filename for standard error (default = <job_name>.e<job_id>)
# at end of job, it is in directory from which qsub was executed


# Export all my environment variables to the job
#PBS -V

#
# set echo               # echo commands before execution; use for debugging
#

# End of the part that shouldn't be changed

# ***  --- USer Changes start here ----
#
#
#
# 1. The requested wall clock job time limit in HH:MM:SS
#    Your job will end when it exceeds this time limit
#
#
#PBS -l walltime=00:01:00

#
# 2. *** Job name
#
# job name (default = name of script file)
# Change at will
#PBS -N FFT_test
#
#

#
# 3.  Your email address
#
#  ***********************
#  !!!!!!!!!!!!!!!!!!!!!!!!
#
# Change to the address you want the notification sent to!!
#PBS -M kaatish@ucsd.edu
#
#  !!!!!!!!!!!!!!!!!!!!!!!!
#  ***********************
#
#
# Mail is sent when the job terminates normally or abnormally
# Add the letter 'b' to the end of the next line if you also want
# to be notified when the job begins to run
#PBS -m aeb
#

#
# End of embedded QSUB options



# --------- The job starts ---------
# 4.  Put the executeable commands you want to run here
#	It is suggested that you always print out the environment
#	which maybe be useful in troubleshooting
# printenv
date
echo "*** Job Starts"

#  You don't normally want to change this
cd $PBS_O_WORKDIR 	 #change to the working directory

#	It is suggested that you always print out the environment
#	which maybe be useful in troubleshooting
printenv
echo ">>> Job starts"

date

# The commands

echo ""
./fft

echo ""
echo ">>> Job Ends"

date

