* imputeincomeREV4.sas '

* gets total family income measures and imputes missing income values;
* uses marrlyimp rather than marrsepimp;

* Revised February 2008;
* --impute income in a new way;
* --clearly separate spousal from partner income;
* --no longer set sbus to zero when it equals bus;
* --only uses income measures from women who are >=18, living away from home, or mothers;

* Revised May 2008:
* --modifies earned, unearned, and non-taxable income combination measures;

* Revised July 2008;
* --reads in updated wagehr, hrswri, and wksuni measures where missing weeks are accounted for;
* --wagehr and hrswri have been set to missing if more than 20% of weeks are unaccounted for; 

* REV 4: Revised May 2010 (by LJL);
* --uses welfare and UE variables that are cleaned slightly differently from before -- read from 'reextract_welfareUE' data set;
* --slightly different imputations on new welfare/UE measures (set to missing if 'flagged' as bad, valid skips set to missing when earned income is non-missing);
* --also imputes missing welfare/UE measures using regression/average-based imputing the same as other variables;
* --different imputations on all income measures: includes zero values in regression/average-based imputing;
* --drops education benefits from non-taxable income measure;

* NOTE: Drop in sample size in 1985 reflects NLSY dropping military oversample.
*       Drop in sample size in 1991 reflects NLSY dropping poor white oversample.

options pagesize=65 linesize=120 nocenter;

libname tempout 'C:\Data\NLSY Children\Data Creation Programs\Temp2';
libname in4 'C:\Data\NLSY Children\Data Creation Programs';
libname in5 'C:\Data\NLSY Children\Preliminary Data';



*********************************************************************;
* Preliminaries: initial cleaning, deflating, etc.;

data prelimcomp;  set in4.mom; 

*Put zeroes if a variable does not exist for a certain year;

if year>1984 then totinr=0;
if year>1989 then totinas=0;
if year>1988 then totinp=0;
if year=1979 then totinao=0;
if year=1979 then smil=0;
if year=1979 then totinar=0;
if year<1990 then totinpm=0;if year>1993 then totinpm=0;
if year<1990 then totinpw=0;if year>1993 then totinpw=0;
if year<1990 then totinpf=0;if year>1993 then totinpf=0;
if year<1990 then totinap=0;if year>1993 then totinap=0;
if year=1979 then totinv=0;
if year<1982 then totinc=0;
if year<1980 then totino=0; if year>1982 then totino=0;
if year<1980 then totinu=0; if year>1981 then totinu=0;
if year>1982 then wageny=0;
if year<1993 then totincs=0; 
if year>1993 then totina=0; 

*Deflate using Fernando's CPI measure (putting in 1979 dollars);
%include 'C:\Data\NLSY Children\Data Creation Programs\cpitable.sas';

array list   mil smil wage swage bus sbus  
             totamu totamus totina totinar totama 
			 totams totedb totedbs totinr totinro
			 totinas totinp totwea
             totinao  totinpm totinpw totinpf totinap totinv
             totinc totino totinu wageny totincs;

do over list; 
  if list = .B then list=0;                                     * set valid skips to zero;
  if list = .A | list= .C | list= .D | list= .E then list=list; * keep other skip codes;
else list=list/cpily; 
end;

*Rename deflated variables ;
mild=mil; smild=smil; waged=wage; swaged=swage; busd=bus; sbusd=sbus;
totamud=totamu; totamusd=totamus; totinad=totina; totamad=totama; 
totamsd=totams; totedbd=totedb; totedbsd=totedbs; totinrd=totinr;
totinrod=totinro; totinasd=totinas; totinpd=totinp; totwead=totwea;
totinaod=totinao; totinpmd=totinpm; totinpwd=totinpw; totinpfd=totinpf;
totinapd=totinap; totinard=totinar; totinvd=totinv; totincd=totinc;
totinod=totino; totinud=totinu; wagenyd=wageny; totincsd=totincs;



*Set as missing any income component higher than $200,000;
array listd  mild smild waged swaged busd sbusd  
             totamud totamusd totinad totamad 
			 totamsd totedbd totedbsd totinrd totinrod
			 totinasd totinpd totwead totinaod 
             totinpmd totinpwd totinpfd totinapd totinard totinvd
             totincd totinod totinud wagenyd totincsd;

DO over listd ; 
  if listd >200000 then listd=.; 
end;

proc sort; by momid year; run;

* Get marital status last year for separating spousal income from partner income after 1993;
data marrlydat; set in5.vars4incimpute (keep=momid year  marrlyimp);

title1 'Preliminary: marital status last year data';
proc means maxdec=3; 
  var marrlyimp;
  class year;
run;
proc sort; by momid year; run;


* Get new welfare and UE measures (for all years 1979-2000);
data newwelfUE; set in5.reextract_welfareUE;
proc sort; by momid year; run;


* merge married last year data with welfare and UE data;
* --both data sets are for all years;
* --marrlydat includes men and women, newwelfUE only includes women;
* --only keep women;
* --deflate welfare/UE income variables as above;

data marrlywelfUE; 
  merge marrlydat newwelfUE (in=inwelfUE); by momid year;

keep momid year marrlyimp  Rtotwead Rtotamad Rtotamfd Rtotamsd Rtotamud Rtotamusd cpily cpi;

if inwelfUE;

%include 'C:\Data\NLSY Children\Data Creation Programs\cpitable.sas';

array list{6} Rtotwea Rtotama Rtotamf Rtotams Rtotamu Rtotamus;
array listd{6} Rtotwead Rtotamad Rtotamfd Rtotamsd Rtotamud Rtotamusd;

do j=1 to 6; 
  *if list{j} = .B then list=0;      * set all valid skips to zero;
  listd{j}=list{j}/cpily; 
  if listd{j} >200000 then listd{j}=.; 
end;


title1 'Preliminary: marital status last year, new UE, and new welfare data';
  proc means maxdec=3; 
  var marrlyimp Rtotwead Rtotamad Rtotamfd Rtotamsd Rtotamud Rtotamusd;
  class year;
run;


***********************************************************************;
*  create some income components and straighten out spouse/partner distinction;
***********************************************************************;
data components; 
  merge prelimcomp (drop=cpi cpily) marrlywelfUE; by momid year;


* Earned respondent income (includes wages & military sources);

earnincR = waged+mild;

* Earned income for spouse;

earnincS = swaged+smild;

* Set any 'valid skips' for new welfare/UE measures to zero if we have an earnings measure for respondent;

if Rtotwead=.B & earnincR>=0 then Rtotwead=0;
if Rtotamad=.B & earnincR>=0 then Rtotamad=0;
if Rtotamfd=.B & earnincR>=0 then Rtotamfd=0;
if Rtotamsd=.B & earnincR>=0 then Rtotamsd=0;
if Rtotamud=.B & earnincR>=0 then Rtotamud=0;
if Rtotamusd=.B & earnincR>=0 then Rtotamusd=0;


* partner/spouse income dummy;
* -note: not very informative for 1995, 1997, 1999 since only Rtotamusd is available
*        and it is almost always zero or missing, BUT this should not affect income calculations in the end;

partspouse = (swaged>0 | sbusd>0 | smild>0 | Rtotamusd>0 | totedbsd>0); * any income from spouse or partner;
if  (swaged<0 & sbusd<0 & smild<0 & Rtotamusd<0 & totedbsd<0) then partspouse=.;

partner = (partspouse=1 & marrlyimp=0);  * any income from partner;
if partspouse<0 | marrlyimp<0 then partner=.;

spouse = 1-partner;                      * any income from spouse;

* Total income of Partner (opposite sex adult);

if year <=1989 then totincP = totinasd;
if 1990 <= year <=1993 then totincP = totinpmd+totinpwd+totinpfd+totinapd;
if year >=1994 & partner=1 then totincP = swaged+sbusd+smild+Rtotamusd+totinvd+totinrod;  * assume income is for partner if no spouse;
if year>=1994 & partner=0 then totincP = 0;                                              * attribute totinvd and totinrod to partner (man);

* Set various 'spousal' income measures to zero if no spouse for 1994+;
if partner=1 & year >= 1994 then do; * assume income is for partner;
  earnincS = 0; 
  swaged=0;
  sbusd=0;
  smild=0;
  totamusd=0; Rtotamusd=0;
  totedbsd=0;
  totincsd=0;
  totinrod=0;  ******NOW, THIS ONLY INCLUDES RESP + SPOUSE INCOME (NOT PARTNER);
  totinvd=0;   ******NOW, THIS ONLY INCLUDES RESP + SPOUSE INCOME (NOT PARTNER);
end;

* Separate alimony from child support for 1979-82;

if year <= 1982 then do;
  totincd = totinad;  * assume all alimony+child support goes to respondent child support;
  totinad = 0;
end; 

* Assume that whenever totinad=totincd in 1983-93 that it is doublecounting and should be child-support;

if 1983 <= year <= 1993 & totinad=totincd & totinad>0 then totinad=0;

* Create sum of respondent and spousal child support (since they're combined prior to 1994);

childsuppRS = totincd+totincsd;

* Fill in spouse or partner income as zero if other is positive (1990-93);

if partspouse=1 & totinpwd<0 & 1990 <= year <= 1993 then totinpwd=0;
if partspouse=1 & totinpfd<0 & 1990 <= year <= 1993 then totinpfd=0;
if partspouse=1 & totinpmd<0 & 1990 <= year <= 1993 then totinpmd=0;
if partspouse=1 & totinapd<0 & 1990 <= year <= 1993 then totinapd=0;

if (totinpwd>0 | totinpfd>0 | totinpmd>0 | totinapd>0) & swaged<0 & 1990 <= year <= 1993 then swaged=0;
if (totinpwd>0 | totinpfd>0 | totinpmd>0 | totinapd>0) & sbusd<0 & 1990 <= year <= 1993 then sbusd=0;
if (totinpwd>0 | totinpfd>0 | totinpmd>0 | totinapd>0) & smild<0 & 1990 <= year <= 1993 then smild=0;

* some additional components;
wagemild  = waged + mild;
swagemild = swaged + smild;

earnincRS   = earnincR+earnincS;                                             * Resp + spouse (not partner) earned income;
unearnincRS = busd + sbusd + Rtotamud + Rtotamusd + totinrod;                * Resp + spouse (not partner) unearned taxable income excluding alimony, since it's not reported 1994+;
*OLD: nontaxincRS = totinvd + Rtotwead + totedbd + totedbsd + totincd + totincsd;  * Resp + spouse (not partner) non-taxable income;
nontaxincRS = totinvd + Rtotwead + totincd + totincsd;  * Resp + spouse (not partner) non-taxable income;

taxincRS = earnincRS + unearnincRS;
totincRS = earnincRS + unearnincRS + nontaxincRS;

* Make sure any income variables other than UE or welfare measures are re-set to missing for 1995, 1997, 1999;
* -- this is very important!;

array listd2  mild smild waged swaged busd sbusd  
             totamud totamusd totinad totamad 
			 totamsd totedbd totedbsd totinrd totinrod
			 totinasd totinpd totwead totinaod 
             totinpmd totinpwd totinpfd totinapd totinard totinvd
             totincd totinod totinud wagenyd totincsd 
			 totincP childsuppRS
			 wagemild swagemild earnincR earnincS earnincRS unearnincRS nontaxincRS taxincRS totincRS;
DO over listd2; 
  if year in (1995,1997,1999) then listd2=.; 
end;



* PERCENTAGES;
* Sources of income divided by incbase=incresp+incsp+incprtn+incwelf;

%macro chkinc(incvar);
 &incvar.p = .;
 if totincRS>0 then &incvar.p = &incvar / totincRS;

 if &incvar.p=0 then &incvar.posp = .; else &incvar.posp = &incvar.p;
%mend chkinc;

%chkinc(earnincRS);
%chkinc(earnincR);
%chkinc(earnincS);
%chkinc(unearnincRS);
%chkinc(nontaxincRS);

%chkinc(waged);
%chkinc(busd);
%chkinc(mild);

%chkinc(swaged);
%chkinc(sbusd);
%chkinc(smild);

%chkinc(totinad);
%chkinc(totinvd);
%chkinc(totinrod);

%chkinc(totedbd);
%chkinc(totedbsd);
%chkinc(childsuppRS);

%chkinc(totinaod);
%chkinc(totinard);

%chkinc(totinasd);
%chkinc(totinpwd);
%chkinc(totinpmd);
%chkinc(totinpfd);
%chkinc(totinapd);

%chkinc(totamud);
%chkinc(Rtotamud);
%chkinc(totamusd);
%chkinc(Rtotamusd);
%chkinc(totwead);
%chkinc(Rtotwead);

title1 'Cleaned but non-imputed data';
proc means maxdec=2; run;

title2 'Means of components';
proc means maxdec=3;
var momid earnincRS earnincR earnincS unearnincRS nontaxincRS taxincRS totincRS totincP;  
run;
proc means maxdec=4;
var momid waged busd mild swaged sbusd smild totinad totinvd totinrod totedbd totedbsd totincd totincsd childsuppRS
    totinaod totinard totinasd totinpwd totinpmd totinpfd totinapd totwead Rtotwead totamud Rtotamud totamusd Rtotamusd;
run;

title2 'Means for components as a fraction of total income (all values)';
proc means maxdec=3;
var momid earnincRSp earnincRp earnincSp unearnincRSp nontaxincRSp;  
run;
proc means maxdec=4;
var momid wagedp busdp mildp swagedp sbusdp smildp totinadp totinvdp totinrodp totedbdp totedbsdp childsuppRSp
    totinaodp totinardp totinasdp totinpwdp totinpmdp totinpfdp totinapdp totweadp Rtotweadp totamudp Rtotamudp totamusdp Rtotamusdp;
run;

title2 'Means for components as a fraction of total income (only positive values)';
proc means maxdec=3;
var momid earnincRSposp earnincRposp earnincSposp unearnincRSposp nontaxincRSposp;  
run;
proc means maxdec=4;
var momid wagedposp busdposp mildposp swagedposp sbusdposp smildposp totamudposp totamusdposp totinadposp totinvdposp totinrodposp 
    totedbdposp totedbsdposp childsuppRSposp
    totinaodposp totinardposp totinasdposp totinpwdposp totinpmdposp totinpfdposp totinapdposp
    totweadposp Rtotweadposp totamudposp Rtotamudposp totamusdposp Rtotamusdposp;
run;


title2 'Means by year';
proc means maxdec=3;
var earnincRS earnincR earnincS;
class year;
run;
proc means maxdec=3;
var earnincRS unearnincRS nontaxincRS totincRS;
class year;
run;
proc means maxdec=3;
var waged busd mild;
class year;
run;
proc means maxdec=3;
var swaged sbusd smild;
class year;
run;proc means maxdec=3;
var wagemild swagemild;
class year;
run;
proc means maxdec=3;
var totamud Rtotamud totamusd Rtotamusd;
class year;
run;
proc means maxdec=3;
var totinad;
class year;
run;
proc means maxdec=3;
var totinvd totinrod;
class year;
run;
proc means maxdec=3;
var totwead Rtotwead;
class year;
run;
proc means maxdec=3;
var totedbd totedbsd;
class year;
run;
proc means maxdec=3;
var childsuppRS;
class year;
run;
proc means maxdec=3;
var totincP;
class year;
run;

proc means maxdec=2; 
  class year;
run;


**************************************************************************;
*  IMPUTATIONS;
**************************************************************************;

* just keep income component data;
data comptmp;
set components (keep=momid year ageofr cpi cpily 
                  earnincRS earnincR earnincS unearnincRS nontaxincRS taxincRS totincRS childsuppRS totincP
                  mild smild waged wagemild swaged busd sbusd swagemild   
             	  totamud totamusd totinad totamad 
			 	  totamsd totedbd totedbsd totinrd totinrod
			 	  totinasd totinpd totwead totinaod 
             	  totinpmd totinpwd totinpfd totinapd totinard totinvd
				  Rtotwead Rtotamad Rtotamfd Rtotamsd Rtotamud Rtotamusd);

proc sort; by momid; run;

* Need to fill in missing ages when person is not surveyed;

data age79dat; set comptmp (keep=momid year ageofr where=(year=1979));
keep momid age79;
age79=ageofr;


* Final full panel data before any income imputations;

data tempout.compreg; 
  merge comptmp age79dat; by momid;

ageofrimp=ageofr;
if ageofrimp<0 then ageofrimp=age79+year-1979;

title1 'Checking age imputations';
proc means; 
var momid ageofr ageofrimp; 
class year; 
run;

proc sort; by momid year; run;



* A) Impute main income variables for respondent/spouse/partner using individual specific regressions of; 
*    income on age and age-squared (must have at least 8 obs, and persons ages 22+);

%macro impmaininc(incvar);

data regincmain (keep=momid year ageofrimp ageofrimpsq &incvar); set tempout.compreg;

*Delete observations where age and income of respondent is missing;
if ageofrimp<=0 | &incvar<0 then delete;  * changed from REV2 & REV3 versions;

* Delete observations for very young ages;
if ageofrimp<22 then delete;

ageofrimpsq=ageofrimp*ageofrimp;  * quadratic in age;


*Regress income measure on age and age squared (considering only positive values) by momid;
proc reg outest=cincreg;
model &incvar = ageofrimp ageofrimpsq /noprint edf;
by momid;
run;

*Rename the coefficients and save results in cincresp ;
data cincreg2 (drop= _MODEL_ _TYPE_ _RMSE_ _RSQ_ _IN_);  set cincreg;
rename ageofrimp=coefage;
rename ageofrimpsq=coefagesq;


*Merge regression results and compreg, saving data;
data tempout.&incvar.dat (keep=momid year ageofrimp ageofrimpsq &incvar numobs
                     &incvar.fit &incvar.imp intercept coefage coefagesq);
merge  cincreg2 tempout.compreg; by momid;

ageofrimpsq=ageofrimp*ageofrimp;

*Calculate the number of oservation on each individual-specific regression;
numobs= _p_ + _edf_;

*Create a variable with the fitted values;
&incvar.fit = intercept + ageofrimp*coefage + ageofrimpsq*coefagesq;

*Impute only if 8 or more observations and income measure is missing;

if numobs>=8 & &incvar <0 then &incvar.imp = &incvar.fit;
else &incvar.imp = &incvar;

* --if the imputed vaue is negative and true value is missing, set it to zero;
if numobs>=8 & &incvar<0 & -25000<&incvar.fit<0 then &incvar.imp=0;
if &incvar.imp <= -25000 then &incvar.imp=.;

* --do not impute for very young ages;
if ageofrimp<22 & &incvar<0 then &incvar.imp=&incvar;

Title "Income measure &incvar (imputed using quadratic regression if numobs at least 8 and age>21)";
proc means maxdec=1; run;

proc freq;
 tables numobs;
run;

proc means maxdec=3;
 var momid &incvar &incvar.imp;
 class year;
run;

proc sort; by momid year; run;

%mend impmaininc;

%impmaininc(waged);
%impmaininc(mild);
%impmaininc(wagemild);

%impmaininc(swaged);
%impmaininc(smild);
%impmaininc(swagemild);


* B) Impute various income variables for respondent/spouse/partner using individual specific regressions of; 
*    income on age (must have at least 6 obs, and ages 22+);

%macro implininc(incvar);

data regincmain (keep=momid year ageofrimp &incvar); set tempout.compreg;

*Delete observations where age and income of respondent is missing;
if ageofrimp<=0 | &incvar<0 then delete;  * changed from REV2 & REV3 versions;

* Delete observations for very young ages;
if ageofrimp<22 then delete;

*Regress income measure on age (considering only positive values) by momid;
proc reg outest=cincreg;
  model &incvar = ageofrimp /noprint edf;
  by momid;
run;

*Rename the coefficients and save results in cincresp ;
data cincreg2 (drop= _MODEL_ _TYPE_ _RMSE_ _RSQ_ _IN_);  set cincreg;
rename ageofrimp=coefage;

*Merge regression results and compreg, saving data;
data tempout.&incvar.dat (keep=momid year ageofrimp &incvar numobs
                     &incvar.fit &incvar.imp intercept coefage);
merge  cincreg2 tempout.compreg; by momid;

*Calculate the number of oservation on each individual-specific regression;
numobs= _p_ + _edf_;

*Create a variable with the fitted values;
&incvar.fit = intercept + ageofrimp*coefage;

*Impute only if 6 or more observations and income measure is missing;

if numobs>=6 & &incvar <0 then &incvar.imp = &incvar.fit;
else &incvar.imp = &incvar;

* --if the imputed vaue is negative and true value is missing, set it to zero;
if numobs>=6 & &incvar<0 & -25000<&incvar.fit<0 then &incvar.imp=0;
if &incvar.imp <= -25000 then &incvar.imp=.;

* --do not impute for very young ages;
if ageofrimp<22 & &incvar<0 then &incvar.imp=&incvar;

Title "Income measure &incvar (imputed with linear regression if numobs at least 6 and age>21)";
proc means maxdec=1; run;

proc freq;
 tables numobs;
run;

proc means maxdec=3;
 var momid &incvar &incvar.imp;
 class year;
run;

proc sort; by momid year; run;

%mend implininc;

%implininc(busd);
%implininc(sbusd);
%implininc(totinrod);
%implininc(totincP);


* C) Impute various income variables for respondent/spouse/partner using individual specific means ;
*    (must have at least 4 positive obs, and ages 22+);

%macro impmninc(incvar);

data mnmain (keep=momid year ageofrimp &incvar); set tempout.compreg;

*Delete observations where age and income of respondent is missing;
if ageofrimp<=0 | &incvar<0 then delete;  * changed from REV2 & REV3 versions;

* Delete observations for very young ages;
if ageofrimp<22 then delete;

*Calculate individual specific mean of income (considering only non-missing values) and save it;
proc means noprint;
  var &incvar;
  by momid;
  output out=mndata mean=m&incvar; 
run;

*Merge means and compreg. Save data;
data tempout.&incvar.dat (keep=momid year &incvar &incvar.imp m&incvar ageofrimp _freq_) ;
  merge mndata tempout.compreg; by momid;

*Impute if at least 4 observations in each ind. specific mean and missing income measure;
if _freq_ >=4 & &incvar<0 then &incvar.imp = m&incvar;
else &incvar.imp = &incvar;

*If the mean was missing (i.e. all are missing or zero), then impute a zero;
if m&incvar=<0 & &incvar.imp<0 then &incvar.imp=0;

* --do not impute for very young ages;
if ageofrimp<22 & &incvar<0 then &incvar.imp=&incvar;

Title "Income measure &incvar (imputed using means if numobs at least 4 and age>21)";
proc means maxdec=1;  run;

proc freq;
 tables _freq_;
run;

proc means maxdec=3;
 var momid &incvar &incvar.imp;
 class year;
run;

proc sort; by momid year; run;

%mend impmninc;

%impmninc(totamud);
%impmninc(totamusd);
%impmninc(totinvd);
%impmninc(totwead);
%impmninc(totedbd);
%impmninc(totedbsd);
%impmninc(childsuppRS);

%impmninc(Rtotamud);
%impmninc(Rtotamusd);
%impmninc(Rtotwead);



* D) Impute income measures for missing survey years (1995, 1997, 1999) using mean of adjacent years;
*    Only keep data from desired year;

%macro impmissinc(incvar,yrm1,yr,yrp1);

data mnmain (keep=momid year &incvar); set tempout.compreg;

*Only keep desired adjacent years for computing means;
if year=&yrm1 | year=&yrp1;

*Calculate individual specific mean of income from adjacent years and save it;
proc means noprint;
  var &incvar;
  by momid;
  output out=mndata mean=m&incvar; 
run;

*Merge means and compreg. Save data;
data tmp&incvar.dat (keep=momid year &incvar &incvar.imp&yr) ;
  merge mndata tempout.compreg; by momid;

*Impute if at least 1 neighboring observation and missing income measure;
if year=&yr & _freq_ >=1 & &incvar<0 then &incvar.imp&yr = m&incvar;
else &incvar.imp&yr = &incvar;


data tempout.&incvar.&yr.dat; set tmp&incvar.dat (where=(year=&yr));
keep momid &incvar.imp&yr;

proc means maxdec=1;
  Title "Income measure &incvar for &yr (imputed using mean of adjacent years)";
run;

proc sort; by momid; run;

%mend impmissinc;

%impmissinc(waged,1994,1995,1996);
%impmissinc(swaged,1994,1995,1996);
%impmissinc(busd,1994,1995,1996);
%impmissinc(sbusd,1994,1995,1996);
%impmissinc(mild,1994,1995,1996);
%impmissinc(smild,1994,1995,1996);
%impmissinc(wagemild,1994,1995,1996);
%impmissinc(swagemild,1994,1995,1996);
%impmissinc(totamud,1994,1995,1996);
%impmissinc(totamusd,1994,1995,1996);
%impmissinc(totinvd,1994,1995,1996);
%impmissinc(totinrod,1994,1995,1996);
%impmissinc(totwead,1994,1995,1996);
%impmissinc(totedbd,1994,1995,1996);
%impmissinc(totedbsd,1994,1995,1996);
%impmissinc(childsuppRS,1994,1995,1996);
%impmissinc(totincP,1994,1995,1996);

%impmissinc(Rtotwead,1994,1995,1996);
%impmissinc(Rtotamud,1994,1995,1996);
%impmissinc(Rtotamusd,1994,1995,1996);

%impmissinc(waged,1996,1997,1998);
%impmissinc(swaged,1996,1997,1998);
%impmissinc(busd,1996,1997,1998);
%impmissinc(sbusd,1996,1997,1998);
%impmissinc(mild,1996,1997,1998);
%impmissinc(smild,1996,1997,1998);
%impmissinc(wagemild,1996,1997,1998);
%impmissinc(swagemild,1996,1997,1998);
%impmissinc(totamud,1996,1997,1998);
%impmissinc(totamusd,1996,1997,1998);
%impmissinc(totinvd,1996,1997,1998);
%impmissinc(totinrod,1996,1997,1998);
%impmissinc(totwead,1996,1997,1998);
%impmissinc(totedbd,1996,1997,1998);
%impmissinc(totedbsd,1996,1997,1998);
%impmissinc(childsuppRS,1996,1997,1998);
%impmissinc(totincP,1996,1997,1998);

%impmissinc(Rtotwead,1996,1997,1998);
%impmissinc(Rtotamud,1996,1997,1998);
%impmissinc(Rtotamusd,1996,1997,1998);

%impmissinc(waged,1998,1999,2000);
%impmissinc(swaged,1998,1999,2000);
%impmissinc(busd,1998,1999,2000);
%impmissinc(sbusd,1998,1999,2000);
%impmissinc(mild,1998,1999,2000);
%impmissinc(smild,1998,1999,2000);
%impmissinc(wagemild,1998,1999,2000);
%impmissinc(swagemild,1998,1999,2000);
%impmissinc(totamud,1998,1999,2000);
%impmissinc(totamusd,1998,1999,2000);
%impmissinc(totinvd,1998,1999,2000);
%impmissinc(totinrod,1998,1999,2000);
%impmissinc(totwead,1998,1999,2000);
%impmissinc(totedbd,1998,1999,2000);
%impmissinc(totedbsd,1998,1999,2000);
%impmissinc(childsuppRS,1998,1999,2000);
%impmissinc(totincP,1998,1999,2000);

%impmissinc(Rtotwead,1998,1999,2000);
%impmissinc(Rtotamud,1998,1999,2000);
%impmissinc(Rtotamusd,1998,1999,2000);


* Combine all imputations for 1995, 1997, 1999, and original imputation;
%macro combineyrs(incvar);

data tempout.&incvar.959799;
  merge tempout.&incvar.1995dat tempout.&incvar.1997dat tempout.&incvar.1999dat; by momid;
run;

data tempout.&incvar.dat2;
  merge tempout.&incvar.959799 tempout.&incvar.dat; by momid;
  keep momid year &incvar &incvar.imp &incvar.imp2;

  &incvar.imp2 = &incvar.imp;
  if year=1995 & &incvar<0 then &incvar.imp2 = &incvar.imp1995;
  if year=1997 & &incvar<0 then &incvar.imp2 = &incvar.imp1997;
  if year=1999 & &incvar<0 then &incvar.imp2 = &incvar.imp1999;
 
title "Income measure &incvar incorporating imputations 1 and 2";
proc means maxdec=1; run;

proc sort; by momid year; run;

%mend combineyrs;

%combineyrs(waged);
%combineyrs(swaged);
%combineyrs(busd);
%combineyrs(sbusd);
%combineyrs(mild);
%combineyrs(smild);
%combineyrs(wagemild);
%combineyrs(swagemild);
%combineyrs(totamud);
%combineyrs(totamusd);
%combineyrs(totinvd);
%combineyrs(totinrod);
%combineyrs(totwead);
%combineyrs(totedbd);
%combineyrs(totedbsd);
%combineyrs(childsuppRS);
%combineyrs(totincP);

%combineyrs(Rtotwead);
%combineyrs(Rtotamud);
%combineyrs(Rtotamusd);

******************************************************************;
* Merge all the imputed component values and compute imputed total;
* income from the sum of all components;
******************************************************************;

data tempout.imputedcomp;
merge tempout.wageddat2 tempout.milddat2 tempout.wagemilddat2  tempout.busddat2 
      tempout.swageddat2 tempout.smilddat2 tempout.swagemilddat2 tempout.sbusddat2
      tempout.totamuddat2 tempout.totamusddat2 tempout.totinvddat2 tempout.totinroddat2 
      tempout.totweaddat2 tempout.totedbddat2 tempout.totedbsddat2
      tempout.childsuppRSdat2 tempout.totincPdat2
	  tempout.Rtotamuddat2 tempout.Rtotamusddat2 tempout.Rtotweaddat2;
by momid year;
keep momid year wagedimp mildimp wagemildimp busdimp swagedimp smildimp swagemildimp sbusdimp
     totamudimp totamusdimp totinvdimp totinrodimp
     totweadimp totedbdimp totedbsdimp childsuppRSimp totincPimp
     wagedimp2 mildimp2 wagemildimp2 busdimp2 swagedimp2 smildimp2 swagemildimp2 sbusdimp2
     totamudimp2 totamusdimp2 totinvdimp2 totinrodimp2
     totweadimp2 totedbdimp2 totedbsdimp2 childsuppRSimp2 totincPimp2
	 Rtotamudimp Rtotamusdimp Rtotweadimp Rtotamudimp2 Rtotamusdimp2 Rtotweadimp2;


title1 'All imputed components';

proc means maxdec=3; run;

proc means maxdec=3;
  class year;
run;


************************************************************************************;
* Merge all the imputed values with compreg and combine income measures for TAXSIM
************************************************************************************;

data in4.imputed;
merge  tempout.compreg(in=in1) tempout.imputedcomp in4.numdep; by momid year;
drop samidc higgrc hgcmom age79 ageofr;

if in1;

*Calculate some imputed cumulative income measures (sum of imputed components);

earnincRimp = wagemildimp2;
earnincSimp = swagemildimp2;

earnincRSimp   = earnincRimp + earnincSimp;                                                    * Resp + spouse (not partner) earned income;
unearnincRSimp = busdimp2 + sbusdimp2 + Rtotamudimp2 + Rtotamusdimp2 + totinrodimp2;         * Resp + spouse (not partner) unearned taxable income excluding alimony - use REV UE measures;
* OLD: nontaxincRSimp = totinvdimp2 + Rtotweadimp2 + totedbdimp2 + totedbsdimp2 + childsuppRSimp2;  * Resp + spouse (not partner) non-taxable income - use REV welfare measure;
nontaxincRSimp = totinvdimp2 + Rtotweadimp2 + childsuppRSimp2;                             * Resp + spouse (not partner) non-taxable income - use REV welfare measure;

taxincRSimp = earnincRSimp + unearnincRSimp;
totincRSimp = earnincRSimp + unearnincRSimp + nontaxincRSimp;


* Now, clean up total income measures;

if totincRS=0 then do;
  earnincR=.;
  earnincS=.;
  earnincRS=.;
  unearnincRS=.;
  nontaxincRS=.;
  taxincRS=.;
  totincRS=.;   
end;
if totincRSimp=0 then do;
  earnincRimp=.;
  earnincSimp=.;
  earnincRSimp=.;
  unearnincRSimp=.;
  nontaxincRSimp=.;
  taxincRSimp=.; 
  totincRSimp=.;   
end;

array inclist wagedimp mildimp wagemildimp busdimp swagedimp smildimp swagemildimp sbusdimp
     totamudimp totamusdimp totinvdimp totinrodimp
     totweadimp totedbdimp totedbsdimp childsuppRSimp totincPimp
     earnincRimp earnincSimp earnincRSimp unearnincRSimp nontaxincRSimp taxincRSimp totincRSimp
     wagedimp2 mildimp2 wagemildimp2 busdimp2 swagedimp2 smildimp2 swagemildimp2 sbusdimp2
     totamudimp2 totamusdimp2 totinvdimp2 totinrodimp2
     totweadimp2 totedbdimp2 totedbsdimp2 childsuppRSimp2 totincPimp2
	 Rtotamudimp Rtotamusdimp Rtotweadimp Rtotamudimp2 Rtotamusdimp2 Rtotweadimp2;

DO over inclist ; 
  if inclist >200000 then inclist=.; 
end;




%include 'C:\Data\NLSY Children\Data Creation Programs\labelincvarsREV.sas';

title1 'Means of income variables (full NLSY sample)';
proc means maxdec=1;
var momid year waged mild wagemild busd swaged smild swagemild sbusd
     totamud totamusd totinvd totinrod
     totwead totedbd totedbsd childsuppRS totincP
     earnincR earnincS earnincRS unearnincRS nontaxincRS taxincRS totincRS;
run;
proc means maxdec=1;
var momid year wagedimp2 mildimp2 wagemildimp2 busdimp2 swagedimp2 smildimp2 swagemildimp2 sbusdimp2
     totamudimp2 totamusdimp2 totinvdimp2 totinrodimp2
     totweadimp2 totedbdimp2 totedbsdimp2 childsuppRSimp2 totincPimp2
     earnincRimp earnincSimp earnincRSimp unearnincRSimp nontaxincRSimp taxincRSimp totincRSimp;
run;

proc means maxdec=1;
var momid Rtotweadimp2 Rtotamudimp2 Rtotamusdimp2 Rtotamad Rtotamfd Rtotamsd;
run;

proc means maxdec=1;
var momid year wagemild wagemildimp2 busd busdimp2 swagemild swagemildimp2 sbusd sbusdimp2;
class year;
run;

proc means maxdec=1;
var momid year totamudimp2 totamusdimp2 totinvdimp2 totinrodimp2
     totweadimp2 totedbdimp2 totedbsdimp2 childsuppRSimp2 totincPimp2;
class year;
run;

proc means maxdec=1;
var momid year earnincR earnincRimp earnincS earnincSimp;
class year;
run;


proc means maxdec=1;
var momid year earnincRS earnincRSimp unearnincRS unearnincRSimp 
     nontaxincRS nontaxincRSimp taxincRS taxincRSimp totincRS totincRSimp;
class year;
run;


proc means maxdec=1;
var totamudimp2 Rtotamudimp2 Rtotamud;
class year;
run;
proc means maxdec=1;
var totamusdimp2 Rtotamusdimp2 Rtotamusd;
class year;
run;
proc means maxdec=1;
var totweadimp2 Rtotweadimp2 Rtotwead;
class year;
run;
proc means maxdec=1;
var Rtotamad Rtotamfd Rtotamsd;
class year;
run;


proc corr; 
  var totamud Rtotamud;
run;
proc corr; 
var totamudimp2 Rtotamudimp2 Rtotamud;
run;
proc corr; 
  var totamusd Rtotamusd;
run;
proc corr; 
var totamusdimp2 Rtotamusdimp2 Rtotamusd;
run;
proc corr; 
  var totwead Rtotwead;
run;
proc corr; 
var totweadimp2 Rtotweadimp2 Rtotwead;
run;


title1 'Means of all variables (full NLSY sample)';
proc means maxdec=2; run;


data tmpzz; set in4.imputed;

if totwead>0 then totwead=.P;
if Rtotwead>0 then Rtotwead=.P;

title1 'Checking joint distribution for final data';
title2 'Sets positive values to .P';
proc freq;
  tables totwead Rtotwead totwead*Rtotwead /missing;
run;
