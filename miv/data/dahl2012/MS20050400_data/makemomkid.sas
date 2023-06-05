*makemomkid.sas;
*  from BLAUFIN2000.sas of Marina & Fernando;

* Create mother and kid data sets by mother-year;

* NOTE: three measures of mother's highest grade completed: ;
*       (1) higgrc - highest grade completed as of May 1 that year (mother's file);
*       (2) hgcmom - highest grade completed as of survey date (mother's file);
*       (3) hgcm   - highest grade completed (from child's file);

* NOTE: Updated to adjust hours worked by mother by weeks unaccounted for (July 25, 2008);

options pagesize=59 linesize=120 nocenter;

libname in 'C:\Data\NLSY Children\Preliminary Data';
libname new 'C:\Data\NLSY Children\Data Creation Programs';

* first read in Gordon's new measure of hgc for mother;

data momhgc; set in.momhgc;
keep parentid hgcmom79-hgcmom100;

parentid=momid;

array hgcarr1 hgcmom79-hgcmom100;
array hgcarr2 hgc79-hgc100;

do i=1 to 22;
 hgcarr1{i} = hgcarr2{i};  * rename hgc* to hgcmom*;
end;

title1 'Mom educ data';
proc means maxdec=3; run;

proc sort; by parentid; run;

* Mom data set (from NLSY79);

data mom1(keep= momid tnfia tnfi year wage afqt sampid datofbm79 
             datofby79  hgcbym79 hgcbyf79 race 
             forbornf forbornm rural14 numofs79 ageofr higgrc 
			 forborn marstc livemo14 livefa14 liveboth 
			 mil smil wage swage bus sbus samidc  hrswri  
             totamu totamus totina totinar totama 
			 totams totedb totedbs totinr totinro
			 totinas totinp totwea
             totinao totinpm totinpw totinpf totinap totinv
             totinc totino totinu wageny
             rltpep totamf totincs hgcmom  wksuni);   * NEW: added wksuni;

merge in.renincome  in.mothperm in.rennumchil in.renmothyearch
      in.renhoursworked in.incomecomponents2 momhgc;
by parentid;

* permanent measures;

rename proafp81=afqt;

*Keep if female;

if sexofr=2;
momid=parentid;

* country of birth for respondent and her parents;

if cntofb79=1 then forborn=0; else if cntofb79=2 then forborn=1; 
else forborn=.;

if bplrsf79=1 then forbornf=0; else if bplrsf79=2 then forbornf=1; 
else forbornf=.;

if bplrsm79=1 then forbornm=0; else if bplrsm79=2 then forbornm=1; 
else forbornm=.; 

*HGC for mother and father of R;
if hgcbym79=95 then hgcbym79=.;  
if hgcbyf79=95 then hgcbyf79=.; 

* urban/rural at age 14;
if arerea79=1 then rural14=0; else if arerea79=2 then rural14=1; 
else if arerea79=3 then rural14=1; else rural14=.;

* who did respondent live with at age 14?;
if witwhd=11 | witwhd=21 | witwhd=31 |witwhd=41 | witwhd=51 |witwhd=91 then livemo14=1; 
else if witwhd=.A|witwhd=.B|witwhd=.C|witwhd=.D|witwhd=.E then livemo14=.; 
else livemo14=0;

if witwhd=11 | witwhd=12 | witwhd=13 |witwhd=14 | witwhd=15 | witwhd=19 then livefa14=1; 
else if witwhd=.A|witwhd=.B|witwhd=.C|witwhd=.D|witwhd=.E then livefa14=.; 
else livefa14=0;

if witwhd=11 then liveboth=1; 
else if witwhd=.A|witwhd=.B|witwhd=.C|witwhd=.D|witwhd=.E then liveboth=.;
else liveboth=0;

* fill in missing higgrc (highest grade completed by child's mother as of May 1);

if higgrc79>20 | higgrc79=.A | higgrc79=.B | higgrc79=.C | higgrc79=.D | higgrc79=.E then higgrc79=.;
if higgrc100>20 | higgrc100=.A | higgrc100=.B | higgrc100=.C | higgrc100=.D | higgrc100=.E then higgrc100=.;
higgrc95=.; higgrc97=.; higgrc99=.;

array hgcarr higgrc79-higgrc100;

*put 'O ' momid higgrc79-higgrc100;

do i=1 to 22;
  if hgcarr{i}>20 | hgcarr{i}=.A | hgcarr{i}=.B | hgcarr{i}=.C | hgcarr{i}=.D | hgcarr{i}=.E then hgcarr{i}=.;
end;
do i=2 to 21;
  if hgcarr{i}=. & hgcarr{i-1} ne . & hgcarr{i+1} ne . & (hgcarr{i+1}>hgcarr{i-1}+1) then hgcarr{i}=hgcarr{i+1}-1;
  if hgcarr{i}=. & hgcarr{i-1} ne . & hgcarr{i+1} ne . & (hgcarr{i+1}<=hgcarr{i-1}+1) then hgcarr{i}=hgcarr{i+1};
  if hgcarr{i}=. & hgcarr{i-1} = . & hgcarr{i+1} ne .  then hgcarr{i}=hgcarr{i+1};
  if hgcarr{i}=. & hgcarr{i-1} ne . & hgcarr{i+1} = .  then hgcarr{i}=hgcarr{i-1};
end;

if higgrc79=. & higgrc80<=12 then higgrc79=higgrc80-1;
if higgrc79=. & higgrc80>12 then higgrc79=higgrc80;
if higgrc100=. then higgrc100=higgrc99;

*put 'N ' momid higgrc79-higgrc100;

* fill in missing hgcmom (highest grade completed by child's mother at time of her survey);

if hgcmom79>20 | hgcmom79=.A | hgcmom79=.B | hgcmom79=.C | hgcmom79=.D | hgcmom79=.E then hgcmom79=.;
if hgcmom100>20 | hgcmom100=.A | hgcmom100=.B | hgcmom100=.C | hgcmom100=.D | hgcmom100=.E then hgcmom100=.;
hgcmom95=.; hgcmom97=.; hgcmom99=.;

array hgcarr2 hgcmom79-hgcmom100;

put 'O ' momid hgcmom79-hgcmom100;

do i=1 to 22;
  if hgcarr2{i}>20 | hgcarr2{i}=.A | hgcarr2{i}=.B | hgcarr2{i}=.C | hgcarr2{i}=.D | hgcarr2{i}=.E then hgcarr2{i}=.;
end;
do i=2 to 21;
  if hgcarr2{i}=. & hgcarr2{i-1} ne . & hgcarr2{i+1} ne . & (hgcarr2{i+1}>hgcarr2{i-1}+1) then hgcarr2{i}=hgcarr2{i+1}-1;
  if hgcarr2{i}=. & hgcarr2{i-1} ne . & hgcarr2{i+1} ne . & (hgcarr2{i+1}<=hgcarr2{i-1}+1) then hgcarr2{i}=hgcarr2{i+1};
  if hgcarr2{i}=. & hgcarr2{i-1} = . & hgcarr2{i+1} ne .  then hgcarr2{i}=hgcarr2{i+1};
  if hgcarr2{i}=. & hgcarr2{i-1} ne . & hgcarr2{i+1} = .  then hgcarr2{i}=hgcarr2{i-1};
end;

if hgcmom79=. & hgcmom80<=12 then hgcmom79=hgcmom80-1;
if hgcmom79=. & hgcmom80>12 then hgcmom79=hgcmom80;
if hgcmom100=. then hgcmom100=hgcmom99;

put 'N ' momid hgcmom79-hgcmom100;

*Stack years on top of each other including annual measures;

%macro panel1(styr=,endyr=);
%do yr=&styr %to &endyr;
  if &yr=100 then year=2000; else year=19&yr;
  if &yr<=86 then tnfia=tnfia&yr; else tnfia=tnfib&yr;
  tnfi=tnfib&yr;
  wage=wage&yr;
  ageofr=ageofr&yr;
  higgrc=higgrc&yr;
  marstc=marstc&yr;
  mil=mil&yr;
  smil=smil&yr;
  swage=swage&yr;
  bus=bus&yr; 
  sbus=sbus&yr;
  hrswri=hrswri&yr;
  wksuni=wksuni&yr;  ***NEW: added wksuni;

  totamu=totamu&yr;
  totamus=totamus&yr;
  totina=totina&yr;
  totinar=totinar&yr;
  totama=totama&yr;
  *totamf=totamf&yr;
  totams=totams&yr;
  totedb=totedb&yr;
  totedbs=totedbs&yr;
  totinr=totinr&yr;
  totinro=totinro&yr;
  totinas=totinas&yr;
  totinp=totinp&yr;
  totwea=totwea&yr;
  totinao=totinao&yr;
  
  totinpm=totinpm&yr;
  totinpw=totinpw&yr;
  totinpf=totinpf&yr;
  totinap=totinap&yr;
  totinv=totinv&yr;

  totinc=totinc&yr;
  totino=totino&yr;
  totinu=totinu&yr;
  wageny=wageny&yr;

  rltpep=rltpep&yr;
  totamf=totamf&yr;
  totincs=totincs&yr;

  hgcmom=hgcmom&yr;

output;
%end;
%mend;
%panel1(styr=79,endyr=94);
%panel1(styr=96,endyr=96);
%panel1(styr=98,endyr=98);
%panel1(styr=100,endyr=100);


data new.mom;   set mom1;
  
if higgrc>20 then higgrc=.;

if marstc=2 then married=1; else if marstc=.A|marstc=.B|marstc=.C|
marstc=.D|marstc=.E then married=.; else married=0;

if marstc=3 then widdivsep=1; else if marstc=.A|marstc=.B|marstc=.C|
marstc=.D|marstc=.E then widdivsep=.; else widdivsep=0;

year86=(year=1986);
year88=(year=1988);
year90=(year=1990);
year92=(year=1992);
year94=(year=1994);
year96=(year=1996);
year98=(year=1998);
year00=(year=2000);

***********************************************************;
* NEW: adjusting hours worked by weeks unaccounted for;

hrswri = hrswri / (1-wksuni/100); * assume missing weeks are just like observed weeks and blow up hours accordingly;
if wksuni > 20 then hrswri=.;     * set hours to missing if more than 20% of all weeks are unaccounted for;

***********************************************************;

wagehr=.;
if hrswri>0 then wagehr=wage/hrswri;


proc sort;  by momid year; run;

title1 'Mom data set';
proc means maxdec=3; run;

proc means maxdec=3;
  var higgrc hgcmom;
  class year;
run;


proc freq;
  tables higgrc hgcmom higgrc*hgcmom;
run;


*****************************************************************************;
* Kids data (will stack by mom-year);

data kid1(keep= momid year piamats piarecs piarers ppvtos 
  idchild male bpitosa motsos homtos black hispanic other 
  ageofc chiage ageofs hgcbys numofa numofc motstg fatstg
  vermab hgcm datofby);

merge in.renhgc in.permchild in.infochild in.renPeabodychild
      in.renBehavior in.renHomescoreschild in.renHousehldchild
      in.Househldchild2 in.renfatmostg in.verbmem ;

by idchild;

* permanent characteristics for child;

rename idmom=momid;

if sexofc=2 then male=0; else if sexofc=1 then male=1; else male=.;

if racofc=2 then black=1; else if racofc=.A| racofc=.B|racofc=.C|
racofc=.D | racofc=. then  black=.; else black=0;

if racofc=1 then hispanic=1; else if racofc=.A| racofc=.B|racofc=.C|
racofc=.D | racofc=. then  hispanic=.; else hispanic =0; 

if racofc=3 then other=1; else if racofc=.A| racofc=.B|racofc=.C|
racofc=.D | racofc=. then  other=.;else other=0; 

* stack by year, including annual outcomes;

%macro panel2(styr=,endyr=);
%do yr=&styr %to &endyr;
  if &yr=100 then  year=2000; 
  else year=19&yr;
  piamats=piamats&yr;
  piarecs=piarecs&yr;
  piarers=piarers&yr;
  ppvtos=ppvtos&yr;
  bpitosa=bpitosa&yr;
  motsos=motsos&yr;
  homtos=homtos&yr;
  chiage=chiage&yr;
  ageofc=ageofc&yr; 
  ageofs=ageofs&yr;
  hgcbys=hgcbys&yr;
  numofa=numofa&yr;
  numofc=numofc&yr;
  motstg=motstg&yr;
  fatstg=fatstg&yr;
  homtos=homtos&yr;
  vermab=vermab&yr;
  hgcm=hgcm&yr;
  output;
%end;
%mend;

%panel2(styr=79,endyr=94);
%panel2(styr=96,endyr=96);
%panel2(styr=98,endyr=98);
%panel2(styr=100,endyr=100);


data new.kid; set kid1;

* clean up missing values for HGC of mother and her spouse (from children of NLSY);

  if hgcm>20 |hgcm<0 | hgcm=.A | hgcm=.B | hgcm=.C | hgcm=.D | hgcm=.E then hgcm=.;
  if hgcbys>20 | hgcbys<0 | hgcbys=.A | hgcbys=.B | hgcbys=.C | hgcbys=.D | hgcbys=.E then hgcbys=.;
  
 /*
 if hgcbys=. then hgcbysmis=1;  else hgcbysmis=0;
 if hgcm=. then hgcmmis=1;  else hgcmmis=0;
*/

proc sort; by momid year; run;

title1 'kid data set';
proc means maxdec=3; run;

proc means maxdec=3;
  var hgcm;
  class year;
run;

proc freq;
  tables hgcm;
run;



data tmp;
  merge new.mom (keep=momid year higgrc hgcmom) new.kid (keep=momid year hgcm); by momid year;

if momid<500 then put momid year higgrc hgcm hgcmom;

diff1 = higgrc-hgcm;
diff2 = higgrc-hgcmom;
diff3 = hgcm-hgcmom;

run;

title1 'Checking education measures';
proc means; 
  var higgrc hgcm hgcmom;
  class year;
run;

proc freq;
  tables higgrc*hgcm hgcm*hgcmom;
run;

proc corr;
  var higgrc hgcm hgcmom;
run;

proc freq; 
  tables diff1 diff2 diff3;
run;

proc sort; by year;
proc freq; by year; 
  tables diff1 diff2 diff3;
run;

