*****Create first-differenced and related variables*****
clear all
set memory 2G
use $data/merged, replace
capture drop _merge

drop if year<1985
*Convert pwages to real 2000$ dollars
replace pwages=(2.295/(10000*cpily))*pwages

***PIAT test variables
replace piamatsn=. if piamats==0
replace piarecsn=. if piarecs==0
replace piarersn=. if piarers==0

gen piaave=(piamatsn+piarersn+piarecsn)/3
sum piaave if samprandom
replace piaave=(piaave-r(mean))/r(sd)

*Shorter PIAT variable names
rename piamatsn math
rename piarersn rer
rename piarecsn rec
rename piaave mathread

sort idchild year
xtset idchild year, yearly delta(1)

*Take second difference of piat variables
local vars "math rer rec mathread"
  foreach var of varlist `vars' {
  gen d02`var' = `var'-L2.`var'
}

gen getdiv13=L1.marrlyimp==0 & (L2.marrlyimp==1|L3.marrlyimp==1)
gen getdiv24=L2.marrlyimp==0 & (L3.marrlyimp==1|L4.marrlyimp==1)
gen getmarr13=L1.marrlyimp==1 & (L2.marrlyimp==0|L3.marrlyimp==0)
gen getmarr24=L2.marrlyimp==1 & (L3.marrlyimp==0|L4.marrlyimp==0)
gen getdiv04=getdiv02|getdiv13|getdiv24
gen getmarr04=getmarr02|getmarr13|getmarr24

***differences with various lags for income variables
gen d02inc012 = inc012 - L2.inc012
gen d13inc012 = L1.inc012 - L3.inc012
gen d24inc012 = L2.inc012 - L4.inc012

***IV variables
gen d02eitcsim211new=eitc211 - L2.eitc011 + stateeitc211 - L2.stateeitc011
gen d02eitcsim311new=eitc311 - L2.eitc011 + stateeitc311 - L2.stateeitc011
gen d02eitcsim411new=eitc411 - L2.eitc011 + stateeitc411 - L2.stateeitc011
gen d13eitcsim411new=L1.eitc611 - L3.eitc711  + L1.stateeitc611 - L2.stateeitc711
gen d24eitcsim411new=L2.eitc011 - L4.eitc511  + L2.stateeitc211 - L4.stateeitc511

***Other variables
gen d02nontax=nontaxincrsimp-L2.nontaxincrsimp
gen d02inc012nontax=d02inc012 + d02nontax
gen d24inc012nontax=(L2.inc012 + L2.nontaxincrsimp) - (L4.inc012 + L4.nontaxincrsimp)
gen d13inc012nontax=(L1.inc012 + L1.nontaxincrsimp) - (L3.inc012 + L3.nontaxincrsimp)
gen d24eitcreal=L2.eitc012 - L4.eitc012

gen dsuminc012nontax=d13inc012nontax+d24inc012nontax
gen dsumeitcsim411new=d13eitcsim411new+d24eitcsim411new
gen inc012nontax=inc012 + nontaxincrsimp

gen part=hrswrinew>0
gen hrs=hrswrinew
gen dpart=part-L2.part
gen dhrs=hrs-L2.hrs

***data restrictions for estimation samples
gen nonpoorsamp=(samprandom&sampnm)|sampnmblack|sampnmhisp
gen flagpretax = (2.295/(10000*cpily))*pretaxinc<10 & (2.295/(10000*L2.cpily))*L2.pretaxinc<10
gen flagtot=abs(d02inc012nontax)<4
gen dearn=earnincrsimp-L2.earnincrsimp
gen drtotwead=(2.295/10000)*(rtotweadimp2-L2.rtotweadimp2)
gen flagdwea=!(drtotwead<-.25 & (dearn<-drtotwead)) & !(drtotwead>.25 & drtotwead<. & (dearn>-drtotwead))
gen estsamp=nonpoorsamp & flagdwea & year>=1989 & !getmarr02 & !getdiv02 & flagtot & flagpretax
quietly ivregress 2sls d02mathread (d02inc012nontax = d02eitcsim411new) x0 x1 x2 x3 x4 x5 $grpbase if estsamp, cluster(momid) first
capture drop esamp
gen esamp0=e(sample)

gen dearn5=(L2.test-L4.test)
gen drtotwead5=(2.295/10000)*(L2.rtotweadimp2-L4.rtotweadimp2)
gen flagdwea5=!(drtotwead5<-.25 & (dearn5<-drtotwead5)) & !(drtotwead5>.25 & drtotwead5<. & (dearn5<-drtotwead5))
gen flag5 = L2.test<10 & L4.test<10 & flagdwea5

gen dearn6=test-L1.test
gen drtotwead6=(2.295/10000)*(rtotweadimp2-L1.rtotweadimp2)
gen flagdwea6=!(drtotwead6<-.25 & (dearn6<-drtotwead6)) & !(drtotwead6>.25 & drtotwead6<. & (dearn6<-drtotwead6))
gen flag6 = year>=1989 & L1.test<10 & test<10 & flagdwea6

gen dearn7=(test-L3.test)
gen drtotwead7=(2.295/10000)*(rtotweadimp2-L3.rtotweadimp2)
gen flagdwea7=!(drtotwead7<-.25 & (dearn7<-drtotwead7)) & !(drtotwead7>.25 & drtotwead7<. & (dearn7<-drtotwead7))
gen flag7 = year>=1989 & L3.test<10 & test<10 & flagdwea7

drop dearn* drtotwead* flagdwea5 flagdwea6 flagdwea7

save $data/firstdiff, replace

