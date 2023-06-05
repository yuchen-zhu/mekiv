*****Stata programs to replicate tables in Dahl and Lochner, "The Impact of Family Income on Child Achievement: Evidence from the Earned Income Tax Credit"
*****Forthcoming in the American Economic Review

*****Note 1: These programs use as inputs main.dta (data from the NLSY) and welfare_data.csv (data on state school accountability and welfare reforms)
*****Note 2: Since the variable for "state" is only available in the restricted use version of the NLSY, in main.dta this variable is set to 0
*****Note 3: Those interested in using information on the state must apply for access to the restricted NLSY and merge this variable into the dataset
*****Note 4: Without the state variable, the results will not match the published results

global prog = "/Users/yuchenzhu/python_projects/ml/projects/measurement-error/miv/data/dahl2012/MS20050400_data"
global data = "/Users/yuchenzhu/python_projects/ml/projects/measurement-error/miv/data/dahl2012/MS20050400_data"
global out = "/Users/yuchenzhu/python_projects/ml/projects/measurement-error/miv/data/dahl2012/MS20050400_data"

*Setup
set more off
clear all
set matsize 800
set memory 2G

*Step 1: Send data to taxsim and get eitc
do $prog/taxsim-eitc.do

*Step 2: Merge in state data for school accountability and welfare reforms
*Note that this only works if you have the "state" variable merged in from the restricted NLSY dataset
do $prog/merge-school-welfare.do

*Step 3: Create first-differenced data and other covariates
do $prog/makevars.do

*Step 4: Run regressions to replicate tables
do $prog/regressions.do
