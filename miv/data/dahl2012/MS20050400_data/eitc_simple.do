*****Prepare data for calculation of EITC, send to TAXSIM, and create dataset for IV

*****Setup

*Baseline without state
**001 is wage income only
**002 is wage plus unearned income
**012, 011 include state taxes and credits

foreach j in 011 {

***KEY to datasets***
*hat=0: real data, hat=4: baseline predicted, hat=2,3 needed for Table 4, hat=5,6,7 needed for Table 5
*state=0: no state tax information calculated, state=1: state taxes included
*spec=1: excludes unearned income, spec2: includes unearned income
local hat=substr("`j'",1,1)
local state=substr("`j'",2,1)
local spec=substr("`j'",3,1)

use $data/main, replace
sort idchild year

*IMPORTANT NOTE: all income variables are in 1979 dollars, and need to be converted to nominal dollars before sending through taxsim
*(convert to nominal dollars by multiplying by cpily)

replace earnincrimp=earnincrimp*cpily
replace earnincsimp=earnincsimp*cpily
replace earnincrsimp=earnincrsimp*cpily
replace unearnincrsimp=unearnincrsimp*cpily
replace nontaxincrsimp=nontaxincrsimp*cpily
gen pretaxinc=earnincrsimp+unearnincrsimp+nontaxincrsimp

*****As needed, make variable changes or sample restrictions

*note: one year already subtracted off to reflect that NLSY reports lagged income
gen trend=year-1986

***Set state variable to missing if necessary (for state=0 samples)
replace state=0 if `state'==0

*****Sample restrictions
gen samp=1
replace samp = (year==1985|year==1987|year==1989|year==1991|year==1993|year==1995|year==1997|year==1999) & (piamatsn!=. | piarecsn!=. | piarersn!=.) if `hat'==2 | `hat'==3 | `hat'==4 | `hat'==5
replace samp = samp | (year==1988|year==1990|year==1992|year==1994|year==1996|year==1998) if `hat'==6 | `hat'==7

sort idchild year
xtset idchild year, yearly delta(1)

*create variables for SIV and to define sample
gen test=(2.295/(10000*cpily))*earnincrsimp
gen x0=L2.test==0
gen x1=L2.test
gen x2=x1^2
gen x3=x1^3
gen x4=x1^4
gen x5=x1^5

gen getdiv02=marrlyimp==0 & (L1.marrlyimp==1|L2.marrlyimp==1)
gen getmarr02=marrlyimp==1 & (L1.marrlyimp==0|L2.marrlyimp==0)

gen flag1 = (2.295/(10000*cpily))*pretaxinc<10 & (2.295/(10000*L2.cpily))*L2.pretaxinc<10
gen flag = year>=1989 & !getmarr02 & !getdiv02 & flag1 if `hat'==2 | `hat'==3 | `hat'==4 | `hat'==6 | `hat'==7
replace flag = year>=1987 & !getmarr02 & !getdiv02 & flag1 if `hat'==5

keep if samp

*drop poor oversample
keep if ((samprandom&sampnm)|sampnmblack|sampnmhisp) | (year==1994|year==1996|year==1998)

*****Get predicted income

*Need to have earned income (for eitc credit) and unearned income (for location on eitc schedule)
gen earninc=.
gen unearninc=.

*Use actual data (for hat=0 samples)
replace earninc=earnincrsimp if `hat'==0
replace unearninc=unearnincrsimp if `hat'==0

*year trend
if `hat'==2 {
  reg test x0 x1 x2 x3 x4 x5 trend if samp & flag
  predict earninchat if samp & flag
  replace earninc=earninchat*((10000*cpily)/2.295) if samp & flag
}

*year dummies
if `hat'==3 {
  reg test x0 x1 x2 x3 x4 x5 yy* if samp & flag
  predict earninchat if samp & flag
  replace earninc=earninchat*((10000*cpily)/2.295) if samp & flag
}

*baseline
if `hat'==4 {
  reg test x0 x1 x2 x3 x4 x5 if samp & flag
  predict earninchat if samp & flag
  replace earninc=earninchat*((10000*cpily)/2.295) if samp & flag
}

*for Table 5
if `hat'==5 {
  gen f0=F2.test==0
  gen f1=F2.test
  gen f2=f1^2
  gen f3=f1^3
  gen f4=f1^4
  gen f5=f1^5

  reg test f0 f1 f2 f3 f4 f5 if samp & flag
  predict earninchat if samp & flag
  replace earninc=earninchat*((10000*cpily)/2.295) if samp & flag
}

*for Table 5
if `hat'==6 {
  gen f0=L1.test==0
  gen f1=L1.test
  gen f2=f1^2
  gen f3=f1^3
  gen f4=f1^4
  gen f5=f1^5

  reg test f0 f1 f2 f3 f4 f5 if samp & flag & (year==1987|year==1989|year==1991|year==1993|year==1995|year==1997|year==1999) & (piamatsn!=. | piarecsn!=. | piarersn!=.)
  predict earninchat if samp & flag 
  replace earninc=earninchat*((10000*cpily)/2.295) if samp & flag 
}

*for Table 5
if `hat'==7 {
  gen f0=F1.test==0
  gen f1=F1.test
  gen f2=f1^2
  gen f3=f1^3
  gen f4=f1^4
  gen f5=f1^5
  
  reg test f0 f1 f2 f3 f4 f5 if samp & flag & (year==1987|year==1989|year==1991|year==1993|year==1995|year==1997|year==1999) & (piamatsn!=. | piarecsn!=. | piarersn!=.)
  predict earninchat if samp & flag
  replace earninc=earninchat*((10000*cpily)/2.295) if samp & flag
}

gen oldyear=year

*****Replace pwages and ui with predicted values earninc and unearninc and save pre-eitc dataset

replace swages=0
replace pwages=earninc
replace ui=unearninc

*Set non-wage income (unearnincrsimp, which we are feeding through taxsim as "ui") equal to zero if necessary (for spec=1 samples)
replace ui=0 if `spec'==1
replace ui=0 if ui==.

replace pwages=0 if pwages==.

drop earninc unearninc

*Save pre-eitc master dataset, using real data, for later merging in future program
if "`j'"=="012" {
  save $data/preeitcinput, replace
}

keep idchild cpily year state mstat depx agex pwages swages dividends otherprop pensions gssi transfers rentpaid proptax otheritem childcare ui depchild mortgage stcg ltcg oldyear

*Note: don't send idchild to taxsim if using state identifiers

*****Send data to TAXSIM and get after tax income and EITC

taxsim35, replace full

replace year=oldyear

sort idchild year
save $data/taxsimout`j', replace

}

*****Now rename taxsim variables and merge taxsim datasets together

*****Rename variables from taxsim output

// local datasets 012 011 211 311 411 511 611 711

// foreach i of local datasets {
//   use $data/taxsimout`i', replace
//   rename v10 fedagi
//   rename v25 eitc
//   rename v28 fedinctax
//   rename v39 stateeitc
//   rename v40 statetotcredit
//   keep idchild year fedagi eitc fedinctax stateeitc statetotcredit siitax cpily pwages otherprop

//   *Put dollar values back in real terms (2000 dollars, so divide by cpily and multiply by 2.295) and divide all monetary variables by 10,000
//   local vars "fedagi eitc fedinctax stateeitc statetotcredit siitax pwages"
//     foreach var of varlist `vars' {
//     qui replace `var' = `var'*(2.295/(10000*cpily))
//   }

//   *Create income variables
//   gen inc=fedagi-fedinctax-siitax+statetotcredit
//   gen incnotax=pwages+eitc
//   gen tax=fedagi-eitc

//   *rename variables with appropriate extension
//   local vars "fedagi eitc fedinctax stateeitc statetotcredit siitax pwages inc incnotax tax"
//   foreach var of varlist `vars' {
//     rename `var' `var'`i'
//   }

//   save $data/taxsim`i', replace
// }

// *****Merge all datasets together
// *Start with preeitcinput for real data
// use $data/preeitcinput, replace
// sort idchild year

// foreach dataset of local datasets {
//   merge idchild year using $data/taxsim`dataset'
//   tab _merge
//   drop _merge
//   sort idchild year
// }

// *Now for additional variables, put back in real terms (into 2000 dollars, so divide by cpily and multiply by 2.295) and divide all monetary variables by 10,000
//   local vars "ui otherprop nontaxincrsimp earnincrsimp unearnincrsimp"
//     foreach var of varlist `vars' {
//     replace `var' = `var'*(2.295/(10000*cpily))
//   }

// *Now for some variables, put into year 2000 dollars in real terms (convert from 1979 dollars to 2000 dollars, so multiply by 2.295) and divide all monetary variables by 10,000
//   local vars "totweadimp2"
//     foreach var of varlist `vars' {
//     replace `var' = `var'/(10000*(1/2.295))
//   }

// save $data/taxsim-merged, replace
