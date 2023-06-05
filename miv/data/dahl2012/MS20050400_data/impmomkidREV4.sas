* impmomkidREV4.sas;

* Begins to generate data to conduct analyses;
* merges mom, kid, and imputed data sets to generate imputmomkid with all data;

* Modified Feb 2008 to take into account new income imputations and aggregates;
* Modified July 2010;
*  - to change sample indicators;
*  - fix creation of 'missing education' measures;


options pagesize=59 linesize=120 nocenter;
* options formdlim='*';


libname in 'C:\Data\NLSY Children\Data Creation Programs';
libname in5 'C:\Data\NLSY Children\Preliminary Data';

****************************************************************;
* Merge all mother data;

* imputed income data;
data impdat; set in.imputed;
proc sort; by momid year; run;

title "Imputed data set (income measures)";
proc means; 
var totincRS earnincRS unearnincRS nontaxincRS;
class year; 
run;
proc means; 
var totincRSimp earnincRSimp unearnincRSimp nontaxincRSimp;
class year; 
run;


* Get marital status last year (marrylimp) as used in income imputation program;
data marrlydat; set in5.vars4incimpute (keep=momid year marrlyimp);

title "married last year data";
proc means; class year; run;
proc sort; by momid year; run;


* mom characteristics data;
data momdat; set in.mom;

* Make missing education variables;
%macro edmiss(ed);
&ed.mis = (&ed <0);  
%mend edmiss;

%edmiss(hgcmom);
%edmiss(higgrc);
%edmiss(hgcbyf79);
%edmiss(hgcbym79);

title1 'mom data set';
proc means; run;
proc means; class year; run;

proc sort; by momid year; run;

* mom AFQT scores;
data afqtdat; set in.afqtadj;   * this data is only for one year;
proc sort; by momid; run;


data momdat2; 
  merge momdat(in=inmom) afqtdat; by momid;
if inmom;

proc sort; by momid year; run;


* Merge imputed data with mom data;

data minc04;
  merge  impdat(in=ink) momdat2(in=inj) marrlydat; by momid year;

*if ink & inj;
if ink | inj;


* drop non-deflated income measures that come from mom;

drop mil smil wage swage bus sbus totamu totamus totina totinar totama 
			 totams totedb totedbs totinr totinro totinas totinp totwea
             totinao  totinpm totinpw totinpf totinap totinv
             totinc totino totinu wageny totincs;

* drop imputation versions with mostly missing values for 95, 97, 99;
drop wagedimp mildimp wagemildimp busdimp swagedimp smildimp swagemildimp sbusdimp
     totamudimp totamusdimp totinvdimp totinrodimp
     totweadimp totedbdimp totedbsdimp childsuppRSimp totincPimp;

* Create some different sampling group variables for future analysis (based on mother's sample);

samprandom    = (1 <= samidc<=8);                                 * random sample;
samprandwhite = (samidc=1 | samidc=2 | samidc=5 | samidc=6);      * all non-military whites;
sampnmblack   = (samidc=3 | samidc=7 | samidc=10 | samidc=13);    * all non-military blacks;
sampnmhisp    = (samidc=4 | samidc=8 | samidc=11 | samidc=14);    * all non-military hispanics;
sampnm        = (1 <= samidc <= 14);                              * all non-military;

proc sort; by momid; run;

title1 "Merged imputed, mom, and afqt data";
proc means maxdec=2; run;
proc means; 
var totincRS earnincRS unearnincRS nontaxincRS;
class year; 
run;
proc means; 
var totincRSimp earnincRSimp unearnincRSimp nontaxincRSimp;
class year; 
run;

* Create average income for a mother (by momid);

proc means data=minc04 noprint;
var totincRS  totincRSimp;
by momid;
output out=meansinc04 mean= mtotincRS mtotincRSimp;
run;

data mninc044;
merge minc04 meansinc04; by momid;

proc sort; by momid year; run;


*****************************************************************;
* Get all kids data;

data child01; set in.kid;

* make missing variables for mother/spouse education;
%macro edmissc(ed);
&ed.mis = (&ed <0);  
%mend edmissc;

%edmissc(hgcm);
%edmissc(hgcbys);

title "child data set";
proc means; run;
proc means; class year; run;
proc sort; by idchild year; run;


data testscores; set in.normtestscores;

title "child test score data set";
proc means; class year; run;
proc sort; by idchild year; run;


data impkidage; set in.imputedkidage;
proc sort; by idchild year; run;
title1 'kid age';
proc means; 
  class year;
run;


data child02; 
  merge child01(in=in1) testscores impkidage; by idchild year;

*if in1;

proc sort; by momid year; run;

title1 'child02 data set';
proc means; class year; run;
proc means;
 var idchild;
 class year;
run;

*****************************************************************;
* Merging mothers' and kids' data;

data in.impmomkid;
  merge mninc044 (in=ink) child02 (in=inj); by momid year;
 
if ink & inj;

* Spousal Education/Age "a la Blau" -- interaction of married*education or age of spouse;
if married=0 then do; 
  ageofs=0; hgcbysmis=0; hgcbys=0; 
end;

%macro edmiss2(ed);
if (&ed<0) & (year ne 1995) & (year ne 1997) & (year ne 1999) then &ed=0;
%mend edmiss2;

%edmiss2(hgcm);
%edmiss2(hgcmom);
%edmiss2(higgrc);
%edmiss2(hgcbys);
%edmiss2(hgcbyf79);
%edmiss2(hgcbym79);

title1 'impmomkid data';
proc means; run;
proc means; class year; run;

proc means; 
var totincRS earnincRS unearnincRS nontaxincRS;
class year; 
run;
proc means; 
var totincRSimp earnincRSimp unearnincRSimp nontaxincRSimp;
class year; 
run;

proc means;
  var hgcm hgcmom higgrc hgcbys hgcbyf79 hgcbym79 hgcmmis hgcmommis higgrcmis hgcbysmis hgcbyf79mis hgcbym79mis;
  class year;
run;

*****************************************************************************;
* look at data;
data blaudat; set in.impmomkid;

if samprandom=1; *Choose sample;

* only keep child survey years;
if year=1986 | year=1988 | year=1990 | year=1992 | year=1994 | year=1996 | year=1998 | year=2000;

/*
* Eliminate selected vars with missing (Blau does it in his programs);

array var1 ageofr ageofs ageofc fatstg motstg hgcm hgcbys numofa numofc afqt;

do over var1; 
  if var1=.A | var1=.B | var1=.C |var1=.D| var1=. then delete;
end;

array varsdel black hispanic other male forborn forbornf
              rural14 livemo14 livefa14 liveboth married widdivsep;

do over varsdel; 
  if varsdel=. then delete; 
end;
*/

* Spousal variables;
if married=0 then do; 
  ageofs=0; hgcbysmis=0; hgcbys=0; 
end;

* Only use positive values of income (consistent with Blau's program);
 
IF totincRSimp>0 THEN OUTPUT;

label afqt='AFQT percentile'; 

Title1 "Means for those with positive total income (totincRSimp), kid survey years only, random sample";

proc means; 
var totincRS earnincRS unearnincRS nontaxincRS;
class year; 
run;
proc means; 
var totincRSimp earnincRSimp unearnincRSimp nontaxincRSimp;
class year; 
run;

proc means maxdec=3;
var ageofrimp ageofs ageofc agecsimp fatstg motstg hgcmom higgrc hgcm hgcbys numofa numofc maxndep afqt afqtadj
    black hispanic other male forborn forbornf
    rural14 livemo14 livefa14 liveboth marrlyimp married widdivsep;
class year;
run;

proc means maxdec=3;
var piamatsn piarersn piarecsn ppvtosn bpitosan motsosn homtosn vermabn;
class year;
run;

proc means maxdec=3; 
  var hgcm;
  class hgcmmis;
run;
proc means maxdec=3; run;

