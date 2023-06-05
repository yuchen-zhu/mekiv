* numdepend3.sas;

* creates profile of family--number of children of different ages-- makes numdep.sas7bdat;

options pagesize=59 linesize=120 nocenter;
options formdlim='*';


libname in4 'C:\Data\NLSY Children\Data Creation Programs';
libname in1 'C:\Data\NLSY Children\Preliminary Data';



* 1. Get data on dependents by adding up children of NLSY;

* - requires some manipulation to fill in missing values;

data kid0; set in4.kid (keep= momid idchild year ageofc);
drop ageofc;

agecs=ageofc/12;
proc sort; by idchild year; run;

%macro unstack;
%do i=79 %to 94;
data dat&i; set kid0 (keep=idchild momid year agecs);
keep idchild momid agecs&i;
if year=19&i;
agecs&i = agecs;
run;
%end;
%mend;

%unstack;


data dat96; set kid0 (keep=idchild momid year agecs);
keep idchild momid agecs96;
if year=1996;
agecs96=agecs;
run;

data dat98; set kid0 (keep=idchild momid year agecs);
keep idchild momid agecs98;
if year=1998;
agecs98=agecs;
run;

data dat00; set kid0 (keep=idchild momid year agecs);
keep idchild momid agecs100;
if year=2000;
agecs100=agecs;
run;


data kid0b; 
 merge dat79 dat80 dat81 dat82 dat83 dat84 dat85 dat86 dat87 dat88 dat89 dat90 dat91 dat92 dat93 dat94 dat96 dat98 dat00;
 by idchild;

array agearr agecs79-agecs94 agecs96 agecs98 agecs100; 

agecs95=.; agecs97=.; agecs99=.;

* fill in forward: if a value is missing, take previous one and add one or two;
do j=2 to 16;
  if agearr{j}<0 then agearr{j} = agearr{j-1}+1;
end;

if agecs96<0 then agecs96=agecs94+2;
if agecs98<0 then agecs98=agecs96+2;
if agecs100<0 then agecs100=agecs98+2;

* fill in backward: if a value is missing, take next one and subtract one or two;

if agecs98<0 & agecs100>2 then agecs98=agecs100-2;
if agecs96<0 & agecs98>2 then agecs96=agecs98-2;
if agecs94<0 & agecs96>2 then agecs94=agecs96-2;

do j=1 to 15;
  if agearr{16-j}<0 & agearr{16-j+1}>1 then agearr{16-j} = agearr{16-j+1}-1;
end;

agecs95=agecs94+1;
if agecs95<0 & agecs96>1 then agecs95=agecs96-1;
agecs97=agecs96+1;
if agecs97<0 & agecs98>1 then agecs97=agecs98-1;
agecs99=agecs98+1;
if agecs99<0 & agecs100>1 then agecs99=agecs100-1;

* Data with imputed child's age by idchild-year;
data in4.imputedkidage; set kid0b;
keep idchild year agecsimp momid;

%macro stack;
%do i=79 %to 99;
year=19&i;
agecsimp=agecs&i;
output;
%end;
%mend;

%stack;

year=2000;
agecsimp=agecs100;
output;




data kid1; 
   merge kid0 (keep=idchild year agecs) in4.imputedkidage; by idchild year; 

if idchild<25000 then put idchild momid year agecs agecsimp;

*Children younger than 19 years old;
valid = (0<=agecsimp<19);

*Children younger than 1 years old;
young1=(0<=agecsimp<1);

*Children younger than 6 years old;
young5=(0<=agecsimp<6);

*Children from 6 to 18 years old;
young618=(6<=agecsimp<19);

* children ages 19+;
oldkids=(agecsimp>=19);

title1 'All families from kid data';

proc means maxdec=3;
  var momid agecs agecsimp valid young1 young5 young618 oldkids;
  class year;
run;

proc sort; by momid year; run;

proc means noprint;
var valid young1 young5 young618 oldkids;
by momid year;
output out=numdep sum=numdep young1 young5 young618 oldkids; 
run;

data numdep2; set numdep (drop= _TYPE_ _FREQ_);

proc means maxdec=3; 
 class year;
run;



* 2. Get measures from NLSY mother's file on number of children in HH (created by numchil.sas);

data numofbdat; set in1.rennumchil;
keep momid numofb79-numofb100;
momid=parentid;

title1 'mother data on children';
proc means maxdec=3; run;

data numofbdat2; set numofbdat;
keep numofb year momid;

%macro panel1(styr=,endyr=);
%do yr=&styr %to &endyr;
  if &yr=100 then year=2000; else year=19&yr;
  numofb = numofb&yr;
  if numofb<0 then numofb=.;
  output;
%end;
%mend;
%panel1(styr=79,endyr=94);
%panel1(styr=96,endyr=96);
%panel1(styr=98,endyr=98);
%panel1(styr=100,endyr=100);

year=1995; numofb=.;
output;

year=1997; numofb=.;
output;

year=1999; numofb=.;
output;

proc sort; by momid year; run;


* 3. Get data from Children of NLSY on number of children of mother in HH (created by hhcomp_cnlsy);

data hhcomp; set in1.hhcomp_cnlsy;
keep momid idchild nhhmlt18_79-nhhmlt18_100 nchh79-nchh100;

data hhcomp2; set hhcomp;
keep nhhmlt18 nchh momid idchild year;

%macro panel2(styr=,endyr=);
%do yr=&styr %to &endyr;
  if &yr=100 then year=2000; else year=19&yr;
  nhhmlt18 = nhhmlt18_&yr;
  nchh = nchh&yr;
  output;
%end;
%mend;
%panel2(styr=79,endyr=94);
%panel2(styr=96,endyr=96);
%panel2(styr=98,endyr=98);
%panel2(styr=100,endyr=100);

year=1995; nchh=.; nhhmlt18=.;
output;

year=1997; nchh=.; nhhmlt18=.;
output;

year=1999; nchh=.; nhhmlt18=.;
output;

proc sort; by momid year; run;


* 4. Merge all three dependent data sources and create a 'best' measure;
* NOTE: nchh nearly identical to numofb for mothers;

data numdep3; 
	merge numdep2 numofbdat2 hhcomp2; by momid year;

* first drop every person who doesn't have a child in the sample;
if idchild=. then delete;

*if momid < 1000 then put momid idchild year numdep numofb nchh nhhmlt18;

depcat=.;
if numdep=0 then depcat=0;
if numdep=1 then depcat=1;
if numdep>1 then depcat=2;

chhcat=.;
if nchh=0 then chhcat=0;
if nchh=1 then chhcat=1;
if nchh>1 then chhcat=2;

maxndep = numdep;
if nchh>numdep then maxndep=nchh;

maxdepcat=depcat;
if chhcat>depcat then maxdepcat=chhcat; 

title1 'All data combined';
proc freq;
  tables maxndep numdep numofb nhhmlt18 nchh 
         numdep*numofb numdep*nchh numdep*nhhmlt18 numofb*nhhmlt18 nhhmlt18*nchh numofb*nchh;
run;
proc freq;
  tables maxdepcat depcat chhcat depcat*chhcat;
run;

proc corr;
  var numdep numofb nchh nhhmlt18;
run;

proc means maxdec=3; run;

proc means noprint;
var young1 young5 young618 oldkids numdep nchh numofb maxndep maxdepcat;
by momid year;
output out=numdeptmp mean=young1 young5 young618 oldkids numdep nchh numofb maxndep maxdepcat; 
run;

title1 'final numdep data';

data in4.numdep; set numdeptmp (drop=_type_ _freq_);
drop lmaxndep lmomid;

* make maxndeply (number of children last year);
maxndeply = maxndep-young1;
lmaxndep=lag(maxndep);
lmomid=lag(momid);
*if lmomid=momid & 1980<=year<=1994 then maxndeply=lmaxndep;
if lmomid=momid & year>=1980 then maxndeply=lmaxndep;

if momid<200 then put momid year maxndep maxndeply lmaxndep young1;

proc means maxdec=3; run;

proc means maxdec=3;
  var maxndep maxndeply;
  class year;
run;



* Just look at those where nchh and numdep don't match;

proc sort data=numdep3; by momid idchild year; run;

data kid2; set kid1 (keep=momid idchild year agecsimp);
proc sort; by momid idchild year; run;

data alldep; merge numdep3 kid2; by momid idchild year; 

if nchh ne numdep;
if numdep=0 & nchh=. then delete;

diff1 = nchh-numdep;

title1 'numdep does not match with nchh';
proc freq;
  tables diff1;
run;

proc means maxdec=3; run;

/*
proc print;
 var momid idchild year agecsimp numdep nchh young1 oldkids nhhmlt18;
 run;
*/