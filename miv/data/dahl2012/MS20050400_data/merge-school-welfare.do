set more off
clear all
set memory 2G

use $data/taxsim-merged, replace
sort statefips
save $data/temp, replace

*****merge in state school accountability and welfare info

clear
insheet using $data/welfare_dat.csv
sort statefips
save welfare, replace

merge statefips using $data/temp
drop _merge

drop state
rename statefips state

*accountability and welfare measures
gen conseq=0
replace conseq=1988 if state==7
replace conseq=1991 if state==50
replace conseq=1993 if state==34
replace conseq=1994 if state==44
replace conseq=1995 if state==18
replace conseq=1996 if state==29|state==37|state==43
replace conseq=1997 if state==1|state==31|state==36|state==40|state==49
replace conseq=1998 if state==8|state==22|state==23|state==33|state==47
gen conseqflag=conseq>=1900 & conseq<=2000

gen con88=conseq==1988 & year>=1988+1
gen con91=conseq==1991 & year>=1991+1
gen con93=conseq==1993 & year>=1993+1
gen con94=conseq==1994 & year>=1994+1
gen con95=conseq==1995 & year>=1995+1
gen con96=conseq==1996 & year>=1996+1
gen con97=conseq==1997 & year>=1997+1
gen con98=conseq==1998 & year>=1998+1
gen postconseq=con88|con91|con93|con94|con95|con96|con97|con98

gen report=0
replace report=1993 if state==30
replace report=1994 if state==25
replace report=1995 if state==15|state==17
replace report=1996 if state==24
replace report=1997 if state==9|state==13|state==26|state==42
replace report=1998 if state==27|state==48

gen rep93=report==1993 & year>=1993+1
gen rep94=report==1994 & year>=1994+1
gen rep95=report==1995 & year>=1995+1
gen rep96=report==1996 & year>=1996+1
gen rep97=report==1997 & year>=1997+1
gen rep98=report==1998 & year>=1998+1
gen postreport=rep93|rep94|rep95|rep96|rep97|rep98

gen postaccount=postconseq|postreport

gen san92=sanction_y==1992 & year>=1992+1
gen san93=sanction_y==1993 & year>=1993+1
gen san94=sanction_y==1994 & year>=1994+1
gen san95=sanction_y==1995 & year>=1995+1
gen san96=sanction_y==1996 & year>=1996+1
gen san97=sanction_y==1997 & year>=1997+1
gen san98=sanction_y==1998 & year>=1998+1
gen san99=sanction_y==1999 & year>=1999+1
gen postsan=san92|san93|san94|san95|san96|san97|san98|san99

gen sch92=schreq_y==1992 & year>=1992+1
gen sch93=schreq_y==1993 & year>=1993+1
gen sch94=schreq_y==1994 & year>=1994+1
gen sch95=schreq_y==1995 & year>=1995+1
gen sch96=schreq_y==1996 & year>=1996+1
gen sch97=schreq_y==1997 & year>=1997+1
gen sch98=schreq_y==1998 & year>=1998+1
gen sch99=schreq_y==1999 & year>=1999+1
gen sch04=schreq_y==2004 & year>=2004+1
gen postsch=sch92|sch93|sch94|sch95|sch96|sch97|sch98|sch99|sch04

gen time92=timelimit_y==1992 & year>=1992+1
gen time93=timelimit_y==1993 & year>=1993+1
gen time94=timelimit_y==1994 & year>=1994+1
gen time95=timelimit_y==1995 & year>=1995+1
gen time96=timelimit_y==1996 & year>=1996+1
gen time97=timelimit_y==1997 & year>=1997+1
gen time98=timelimit_y==1998 & year>=1998+1
gen time99=timelimit_y==1999 & year>=1999+1
gen posttime=time92|time93|time94|time95|time96|time97|time98|time99

gen postwelfare=postsan|postsch|posttime

foreach var of varlist x0 x1 x2 x3 x4 x5 {
  gen trend`var' = trend*`var'
  gen postaccount`var'=postaccount*`var'
  gen postwelfare`var'=postwelfare*`var'
}

sort idchild
save $data/merged, replace

