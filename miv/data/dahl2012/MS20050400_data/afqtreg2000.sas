* afqtreg2000;

* makes dataset afqtadj.sas7bdat with momid and the following afqt measures:
*   (1) afqt - afqt in percentile (straight from CD without adjustments);
*   (2) afqtnorm - afqt normalized by mean and std dev of distribution for all females;
*   (3) afqtadj - normalized afqt adjusted for age differences in 1980;


* program for age-adjustments of afqt scores;

options pagesize=59 linesize=120 nocenter;
*options formdlim='*';

libname in 'C:\Data\NLSY Children\Preliminary Data';
libname new 'C:\Data\NLSY Children\Data Creation Programs';


* Effects of a on standardized AFQT measure based on female AFQT distribution;
title1 'Age effects on standardized mother using female distribution';

data afqtA; set new.mom;

* Use a pure random sample (remove oversamples);
if samidc>8 then delete;

* Use data from 1980 (when moms took the AFQT); 
if year=1980;

* Delete observations with ageofr=0; 
if ageofr=0 then delete;

*Eliminate those with missing values of Mother's age and AFQT;

array avar ageofr afqt;
do over avar; if avar=.A | avar=.B |avar=.C |
                 avar=.D| avar=.E| avar=. then delete;
end;

/*
proc means;
  var afqt;
run;
*/

*Construct a standardized measure of AFQT (afqtm=[afqt-mean]/std);
afqtnorm=(afqt-46.5313577)/27.8164606; 


proc means fw=7;
  var afqt afqtnorm;
run;

proc means fw=7;
  var afqt;
  class ageofr;
run;

proc means;
  var afqtnorm;
  class ageofr;
run;

data new.afqtadj; set new.mom (keep=momid ageofr afqt year where=(year=1980));
keep momid afqt afqtnorm afqtadj;

a15=(ageofr=15);
a16=(ageofr=16);
a17=(ageofr=17);
a18=(ageofr=18);
a19=(ageofr=19);
a20=(ageofr=20);
a21=(ageofr=21);
a22=(ageofr=22);
a23=(ageofr=23);

* first normalize to mean zero, std dev of one;
afqtnorm=(afqt-46.5313577)/27.8164606; 

* adjust for age;

afqtadj = afqtnorm - (-.1745615*a15 -.1634594*a16 -.1205973*a17 +.0132094*a18 +.0134239*a19);
afqtadj = afqtadj - (.0650107*a20 +.148088*a21 +.1847602*a22 + .3623178*a23);

if ageofr=. then afqtadj=.;


proc univariate;
  var afqt afqtnorm afqtadj;
run;

