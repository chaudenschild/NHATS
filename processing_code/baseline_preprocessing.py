import numpy as np
import pandas as pd

round_1_sp = pd.read_stata('data/NHATS_Round_1_SP_File.dta')

# Exclusions

# dementia
# (hc1disescn9) - 1 - YES, 2 - NO, -1 Inapplicable, -8 DK, -9 Missing
# filter hc1disescn9 == 2
# n post filter = 7146
round_1_cohort = round_1_sp[round_1_sp.hc1disescn9 == ' 2 NO']

# missing grip strength
# n post filter = 5969
# 2 measures, gr1grp1rdng and gr1grp2rdng
# remove those with na in both measurements
both_grip_na = [pd.isna(g1) and pd.isna(g2) for g1, g2 in zip(
    round_1_cohort.gr1grp1rdng, round_1_cohort.gr1grp2rdng)]
round_1_cohort = round_1_cohort[[not x for x in both_grip_na]]

# missing height or weight data
# weight in pounds (hw1currweigh), height in feet (hw1howtallft), height in inches (hw1howtallin)
# n post filter = 5822
missing_height_or_weight = [any([pd.isna(weight), pd.isna(height_f), pd.isna(height_in)]) for weight, height_f, height_in in zip(
    round_1_cohort.hw1currweigh, round_1_cohort.hw1howtallft, round_1_cohort.hw1howtallin)]
round_1_cohort = round_1_cohort[[not x for x in missing_height_or_weight]]

# Derived measures

# max grip_strength
# max of both grip readings (if applicable)
# appended as max_grip
round_1_cohort['max_grip'] = round_1_cohort.apply(
    lambda x: np.nanmax([x.gr1grp1rdng, x.gr1grp2rdng]), axis=1)

# BMI
# defined as self-reported baseline weight in kg divided by height in meters-squared
# appended as weight_kg, height_m, and BMI respectively
round_1_cohort['weight_kg'] = round_1_cohort.hw1currweigh.astype(
    'float') / 2.2046
round_1_cohort['height_m'] = (round_1_cohort.hw1howtallft.astype(
    'int') * 12 + round_1_cohort.hw1howtallin.astype('int')) * 0.0254
round_1_cohort['bmi'] = round_1_cohort.weight_kg / round_1_cohort.height_m**2

# High waist circumference
# waist measure in inches (wc1wstmsrinc)
# indicator for high waist circumference,  >= 102 cm in males, >= 88 cm in females
# appended as high_wc
# 104 missing wc measure


def high_wc(x):
    if pd.isna(x.r1dgender) or pd.isna(x.wc1wstmsrinc):
        return np.nan
    wc = x.wc1wstmsrinc * 2.54
    if x.r1dgender == '1 MALE':
        return True if wc >= 102 else False
    elif x.r1dgender == '2 FEMALE':
        return True if wc >= 88 else False
    else:
        raise Exception


round_1_cohort['high_wc'] = round_1_cohort.apply(high_wc, axis=1)


# Sarcopenia (defined by grip strength)
# grip strength < 35.5 kg in males, <20 kg in females
# appended as sarcopenia
# no na due to exclusion criteria

def sarcopenia(x):
    if pd.isna(x.max_grip) or pd.isna(x.r1dgender):
        return np.nan
    if x.r1dgender == '1 MALE':
        return True if x.max_grip < 35.5 else False
    elif x.r1dgender == '2 FEMALE':
        return True if x.max_grip < 20 else False
    else:
        raise Exception


round_1_cohort['sarcopenia'] = round_1_cohort.apply(sarcopenia, axis=1)

# SDOC Sarcopenia (defined by grip strength/BMI ratio)
# grip strength/BMI < 1.05 in males, < 0.79 in females
# appended as sdoc_sarcopenia
# no na due to exclusion criteria


def sdoc_sarcopenia(x):
    if any([pd.isna(m) for m in [x.max_grip, x.bmi, x.r1dgender]]):
        return np.nan

    ratio = x.max_grip / x.bmi
    if x.r1dgender == '1 MALE':
        return True if ratio < 1.05 else False
    elif x.r1dgender == '2 FEMALE':
        return True if ratio < 0.79 else False
    else:
        raise Exception


round_1_cohort['sdoc_sarcopenia'] = round_1_cohort.apply(
    sdoc_sarcopenia, axis=1)


# Gender
# r1dgender

round_1_cohort['gender'] = round_1_cohort.r1dgender

# Race
# rl1dracehisp, recode values below in dictionary
# no na
# appended as race


def race(x):
    d = {' 1 White, non-hispanic': 'White', ' 2 Black, non-hispanic': 'Black',
         ' 3 Other (Am Indian/Asian/Native Hawaiian/Pacific Islander/other specify), non-Hispanic': 'Other', ' 4 Hispanic': 'Hispanic', ' 5 more than one DKRF primary': 'Other', ' 6 DKRF': 'DKRF'}

    return d.get(x.rl1dracehisp, np.nan)


round_1_cohort['race'] = round_1_cohort.apply(race, axis=1)


# Smoking status
# Current - sd1smokedreg == 1 (smoked regularly) & sd1smokesnow == 1 (smokes now)
# Former smoker - sd1smokedreg == 1 & sd1smokesnow == 2 or sd1smokesnow is na
# Never - sd1smokedreg == 2 & sd1smokesnow == 2
# appended as smoking_status
# 1 overall na

def smoking_status(x):
    if pd.isna(x.sd1smokedreg) and pd.isna(x.sd1smokesnow):  # only 1
        return np.nan
    elif pd.isna(x.sd1smokedreg) and pd.notna(x.sd1smokesnow):  # never
        raise Exception
    elif pd.notna(x.sd1smokedreg) and pd.isna(x.sd1smokesnow):  # 2818
        if x.sd1smokedreg == ' 1 YES':
            return 'Former, maybe current'
        elif x.sd1smokedreg == ' 2 NO':
            return 'Never'
    else:  # both exist
        if x.sd1smokedreg == ' 1 YES' and x.sd1smokesnow == ' 1 YES':
            return 'Current'
        elif x.sd1smokedreg == ' 1 YES' and x.sd1smokesnow == ' 2 NO':
            return 'Former'
        else:
            return 'Never'


round_1_cohort['smoking_status'] = round_1_cohort.apply(smoking_status, axis=1)

# Education
# el1higstschl
# Less than high school: 1 - no schooling,
#                        2 - 1st to 8th grade,
#                        3 - 9th to 12th grade, no diploma
# High school to some college: 4 - high school graduate (diploma or equivalent)
#                              5 - vocational, technical, business or trade school certificate
#                                       beyond high school
#                              6 - some college but no degree
# College degree: 7 - associate's degree
#                 8 - bachelor's degree
# Graduate degree: 9 - master's, professional, or doctoral
# appended as education
# 4 na


def education(x):

    d = {' 1 NO SCHOOLING COMPLETED': 'Less than high school',
         ' 2 1ST-8TH GRADE': 'Less than high school',
         ' 3 9TH-12TH GRADE (NO DIPLOMA)': 'Less than high school',
         ' 4 HIGH SCHOOL GRADUATE (HIGH SCHOOL DIPLOMA OR EQUIVALENT)': 'High school to some college',
         ' 6 SOME COLLEGE BUT NO DEGREE': 'High school to some college',
         " 8 BACHELOR'S DEGREE": 'College degree',
         " 9 MASTER'S, PROFESSIONAL, OR DOCTORAL DEGREE": 'Graduate degree',
         ' 5 VOCATIONAL, TECHNICAL, BUSINESS, OR TRADE SCHOOL CERTIFICATE OR DIPLOMA (BEYOND HIGH SCHOOL LEVEL)': 'High school to some college',
         " 7 ASSOCIATE'S DEGREE": 'College degree'
         }

    return d.get(x.el1higstschl, np.nan)


round_1_cohort['education'] = round_1_cohort.apply(education, axis=1)

# Physical activity proxy
# pa1evrgowalk
# appended as ever_walk
# no na
round_1_cohort['ever_walk'] = round_1_cohort.apply(
    lambda x: True if x.pa1evrgowalk == ' 1 YES' else False, axis=1)

# Comorbidities

# heart_disease
# hc1disescn1 - had heart attack
# hc1disescn2 - has heart disease
# no na


def heart_disease(x):
    if x.hc1disescn1 == ' 1 YES' or x.hc1disescn2 == ' 1 YES':
        return True
    elif x.hc1disescn1 == ' 2 NO' or x.hc1disescn2 == ' 2 NO':
        return False
    else:
        return np.nan


round_1_cohort['heart_disease'] = round_1_cohort.apply(heart_disease, axis=1)

# hypertension
# hc1disescn3
# 7 na
round_1_cohort['hypertension'] = round_1_cohort.apply(
    lambda x: True if x.hc1disescn3 == ' 1 YES' else False if x.hc1disescn3 == ' 2 NO' else np.nan, axis=1)

# arthritis
# hc1disescn4
# 12 na
round_1_cohort['arthritis'] = round_1_cohort.apply(
    lambda x: True if x.hc1disescn4 == ' 1 YES' else False if x.hc1disescn4 == ' 2 NO' else np.nan, axis=1)

# diabetes
# hc1disescn6
# 2 na
round_1_cohort['diabetes'] = round_1_cohort.apply(
    lambda x: True if x.hc1disescn6 == ' 1 YES' else False if x.hc1disescn6 == ' 2 NO' else np.nan, axis=1)

# lung_disease
# hc1disescn7
# 4 na
round_1_cohort['lung_disease'] = round_1_cohort.apply(
    lambda x: True if x.hc1disescn7 == ' 1 YES' else False if x.hc1disescn7 == ' 2 NO' else np.nan, axis=1)

# stroke
# hc1disescn8
# 5 na
round_1_cohort['stroke'] = round_1_cohort.apply(
    lambda x: True if x.hc1disescn8 == ' 1 YES' else False if x.hc1disescn8 == ' 2 NO' else np.nan, axis=1)

# cancer
# hc1disescn10
# 2 na
round_1_cohort['cancer'] = round_1_cohort.apply(
    lambda x: True if x.hc1disescn10 == ' 1 YES' else False if x.hc1disescn10 == ' 2 NO' else np.nan, axis=1)

# Age category
# r1d2intvrage
# no na
# appended as age_category


def age_category(x):
    d = {
        '1 - 65-69': '65-69',
        '2 - 70-74': '70-74',
        '3 - 75-79': '75-79',
        '4 - 80-84': '80-84',
        '5 - 85-89': '85+',
        '6 - 90 +': '85+'
    }
    return d.get(x.r1d2intvrage, np.nan)


round_1_cohort['age_category'] = round_1_cohort.apply(age_category, axis=1)

# Obesity (defined by BMI)
# BMI >= 30 kg/m^2
# appended as Obesity
# no na due to exclusion criteria

round_1_cohort['obesity'] = round_1_cohort.apply(
    lambda x: True if x.bmi >= 30 else False, axis=1)

# Sarcopenic obesity definitions

# Grouping 1: sarcopenia, obesity, sarcopenic obesity, neither
# obesity derived from BMI (variable obesity)
# appended as grouping_1_so_status


def grouping_1_so_status(x):

    if pd.isna(x.sarcopenia) or pd.isna(x.obesity):  # shouldn't happen
        return np.nan

    if x.sarcopenia and not x.obesity:
        return 'Sarcopenia'
    elif not x.sarcopenia and x.obesity:
        return 'Obesity'
    elif x.sarcopenia and x.obesity:
        return 'Sarcopenic Obesity'
    elif not x.sarcopenia and not x.obesity:
        return 'Neither'


round_1_cohort['grouping_1_so_status'] = round_1_cohort.apply(
    grouping_1_so_status, axis=1)

# Grouping 2: sarcopenia, obesity, sarcopenic obesity, neither
# obesity derived from waist circumference (variable name high_wc)
# appended as grouping_2_so_status
# 104 na (due to missing wc)


def grouping_2_so_status(x):

    if pd.isna(x.sarcopenia) or pd.isna(x.high_wc):
        return np.nan

    if x.sarcopenia and not x.high_wc:
        return 'Sarcopenia'
    elif not x.sarcopenia and x.high_wc:
        return 'Obesity'
    elif x.sarcopenia and x.high_wc:
        return 'Sarcopenic Obesity'
    elif not x.sarcopenia and not x.high_wc:
        return 'Neither'


round_1_cohort['grouping_2_so_status'] = round_1_cohort.apply(
    grouping_2_so_status, axis=1)


# STEADI score
# Adapted from Dr. Matthew Lohman's SAS code

# Giving round option but requires refactoring if used for non baseline rounds
def STEADI_score(x, round):
    # screen 1 conditions
    if all([m in [' 2 NO', np.nan] for m in
            [x[f'hc{round}faleninyr'], x[f'hc{round}worryfall'], x[f'ss{round}prbbalcrd']]]):
        screen1 = 'Low risk'
    elif any([m == ' 1 YES' for m in
              [x[f'hc{round}faleninyr'], x[f'hc{round}worryfall'], x[f'ss{round}prbbalcrd']]]):
        screen1 = 'Medium risk'

    # screen 2 conditions
    screen2 = np.nan
    if screen1 == 'Medium risk':
        if x[f'ba{round}ftdmreslt'] == '2 Attempted, not held for 10 seconds':
            screen2 = 'Potential high risk'
        elif x[f'ba{round}ftdmreslt'] == '3 Not attempted' and any([a == ' 1 Yes' for a in [x[f'ba{round}rsn{i}ftstd'] for i in range(1, 4)]]):
            screen2 = 'Potential high risk'
        elif pd.isna(x[f'ba{round}ftdmreslt']) and x[f'ba{round}dblftadm'] in ['3 Not administered because did not complete prior balance tests', '4 Not Eligible due to exclusion criteria']:
            screen2 = 'Potential high risk'

        # space weirdness from pandas stata read-in
        if x[f'ch{round}2chstrslt'] == '2 Attempted    ':
            screen2 = 'Potential high risk'
        elif x[f'ch{round}2chstrslt'] == '3 Not attempted' and any([a == ' 1 Yes' for a in [x[f'ch{round}chstntat{i}'] for i in range(1, 4)]]):
            screen2 = 'Potential high risk'
        elif pd.isna(x[f'ch{round}2chstrslt']) and x[f'ch{round}drchradm'] in ['3 Not administered because did not complete single chair stand w/o arms', '4 Not Eligible due to exclusion criteria']:
            screen2 = 'Potential high risk'

    # screen 3 conditions
    if screen2 == 'Potential high risk' and (x[f'hc{round}multifall'] == ' 1 YES' or x[f'hc{round}brokebon1'] == ' 1 YES'):
        screen3 = 'High risk'
    else:
        screen3 = np.nan

    # STEADI
    if screen1 == 'Low risk':
        STEADI = 'Low risk'
    elif screen1 == 'Medium risk':
        if pd.isna(screen2) and pd.isna(screen3):
            STEADI = 'Medium risk'
        elif screen2 == 'Potential high risk' and pd.isna(screen3):
            STEADI = 'Medium risk'
        elif screen3 == 'High risk':
            STEADI = 'High risk'

    return STEADI


round_1_cohort['STEADI_score'] = round_1_cohort.apply(
    STEADI_score, round=1, axis=1)


# Cognitive measure derivations
# functions suitable for baseline as well as subsequent rounds
# ref https://www.nhats.org/scripts/documents/NHATS_Addendum_to_Technical_Paper_5_SAS_Programming_Statements_Jul2013.pdf
# AD8 + dementia_class

# alteration from NHATS (flipped scale, NHATS uses 1,2,3 for likely, probable, no dem)
#   dementia class 0: no dementia
#   dementia class 1: probable
#   dementia class 2: likely


def dementia_class(x, round):
    hcdisescn9 = f'hc{round}disescn9'
    rdresid = f'r{round}dresid'
    isresptype = f'is{round}resptype'

    if x[rdresid] in ['3 Residential Care Resident not nursing home (FQ only)', '7 Residential care not nursing home in R1 and R2 (FQ only)']:
        return -9

    # nursing home/ residential care residents, deceased
    elif x[rdresid] in ['4 Nursing Home Resident', '6 Deceased', '8 Nursing home in R1 and R2 (FQ only)']:
        return -1
    # dementia_class = probable if reported by self/proxy
    if x[hcdisescn9] in ['1 YES', '7 PREVIOUSLY REPORTED'] and x[isresptype] in ['1 SAMPLE PERSON (SP)', '2 PROXY']:
        return 1


def ad8_score(x, round):

    def think(item): return f'cp{round}chgthink{item}'
    isresptype = f'is{round}resptype'

    ad8_items = [-1 for i in range(8)]
    if x[isresptype] == '2 PROXY' and pd.isna(x.dementia_class):
        for i in range(8):
            if x[think(i + 1)] in [' 1 YES, A CHANGE', " 3 DEMENTIA/ALZHEIMER'S REPORTED BY PROXY"]:
                ad8_items[i] = 1
            elif x[think(i + 1)] == ' 2 NO, NO CHANGE':
                ad8_items[i] = 0
            else:
                ad8_items[i] = np.nan

    ad8_score = sum(filter(pd.notna, ad8_items))

    if round != 1:  # max ad8 score if ad8 criteria already met in previous round
        if x[f'cp{round}dad8dem'] == '1 DEMENTIA RESPONSE TO ANY AD8 ITEMS IN PRIOR ROUND' and x[isresptype] == '2 PROXY' and pd.isna(x.dementia_class):
            ad8_score = 8

    return ad8_score


def ad8_dementia(x, round):
    if x.ad8_score >= 2:
        return 1
    # if all missing, or score is 0 or 1
    elif x.ad8_score in [0, 1]:
        return 2
    else:
        return np.nan


def update_dementia_class(x, round):
    cgspeaktosp = f'cg{round}speaktosp'
    if pd.isna(x.dementia_class):
        if x.ad8_dementia == 1:  # probable by ad8 criteria
            return 1
        elif x.ad8_dementia == 2 and x[cgspeaktosp] == '2 NO':
            return 0
    else:
        return x.dementia_class


# Orientation domain
# date score and president naming

def orientation_domain(x, round):

    def date_str(item):
        if round == 4 and item == 4:  # fix a miscode in round 4
            return 'cg4todaydat5'
        return f'cg{round}todaydat{item}'

    date_items = [np.nan for _ in range(4)]
    for i in range(4):
        if x[date_str(i + 1)] in [' 1 YES', '1 YES']:
            date_items[i] = 1
        elif x[date_str(i + 1)] in [" 2 NO/DON'T KNOW", '-7 RF']:
            date_items[i] = 0

    if pd.notna(date_items).sum() == 0:
        # Proxy can speak to SP but SP unable to answer
        if x[f'cg{round}speaktosp'] in [' 1 YES', '1 YES']:
            date_sum = 0
        else:
            date_sum = np.nan  # Proxy can't speak to SP
    else:
        date_sum = sum(filter(pd.notna, date_items))

    pres_str = [f'cg{round}presidna1', f'cg{round}presidna3',
                f'cg{round}vpname1', f'cg{round}vpname3']

    pres_items = [np.nan for _ in range(len(pres_str))]
    for i, pres in enumerate(pres_str):
        if x[pres] == ' 1 YES':
            pres_items[i] = 1
        elif x[pres] in [" 2 NO/DON'T KNOW", '-7 RF']:
            pres_items[i] = 0

    if pd.notna(pres_items).sum() == 0:
        # Proxy can speak to SP but SP unable to answer
        if x[f'cg{round}speaktosp'] in [' 1 YES', '1 YES']:
            pres_sum = 0
        else:
            pres_sum = np.nan  # Proxy can't speak to SP
    else:
        pres_sum = sum(filter(pd.notna, pres_items))

    return sum([date_sum, pres_sum])


# Executive domain
# Clock test


def clock_test(x, round):
    cgdclkdraw = f'cg{round}dclkdraw'

    if x[cgdclkdraw] in ['-1 Inapplicable', '-2 Proxy says cannot ask SP'] or pd.isna(x[cgdclkdraw]):
        return np.nan
    elif x[cgdclkdraw] in ['-3 Proxy says can ask SP but SP unable to answer',
                           '-4 SP did not attempt to draw clock',
                           '-7 SP refused to draw clock']:
        return 0

    elif x[cgdclkdraw] == '-9 Missing':  # impute mean score for missing
        if x[f'cg{round}speaktosp'] == '1 YES':
            return 2
        elif x[f'cg{round}speaktosp'] == '-1 Inapplicable':
            return 3
    else:
        return int(x[cgdclkdraw][0])


# Memory domain
# immediate + delayed word recall


def memory_domain(x, round):

    mem_str = [f'cg{round}dwrdimmrc', f'cg{round}dwrddlyrc']

    mem_items = []
    for mem in mem_str:
        if x[mem] in ['-1 Inapplicable', '-2 Proxy says cannot ask SP', '-3 Proxy says can ask SP but SP unable to answer', '-7 SP refused activity', '-9 Missing']:
            mem_items.append(0)
        else:
            # fix a round 2 specific coding error
            if mem == 'cg2dwrdimmrc' and x['cg2dwrdimmrc'] == 10 and x['cg2dwrddlyrc'] == -3:
                x['cg2dwrdimmrc'] = -3
            mem_items.append(x[mem])

    return sum(mem_items)


def domain_binarize(x, round):

    x['clock_binary'] = 0 if 1 < x.clock_test <= 5 else 1 if 0 <= x.clock_test <= 1 else np.nan

    x['memory_binary'] = 0 if 3 < x.memory_domain <= 20 else 1 if 0 <= x.memory_domain <= 3 else np.nan

    x['orientation_binary'] = 0 if 3 < x.orientation_domain <= 8 else 1 if 0 <= x.orientation_domain <= 3 else np.nan

    x['domains_score'] = sum(
        [x['clock_binary'], x['memory_binary'], x['orientation_binary']])

    return x


def final_update_dementia_class(x, round):

    if pd.isna(x['dementia_class']) and x[f'cg{round}speaktosp'] in ['-1 Inapplicable', '1 YES']:
        if x.domains_score in [2, 3]:
            return 2
        elif x.domains_score == 1:
            return 1
        elif x.domains_score == 0:
            return 0
        else:
            return np.nan
    else:
        return x['dementia_class']


def full_cognitive_derivation(sp, round):

    # rename self-rated memory, immediate and delayed recall for convenience
    sp['self_rated_memory'] = sp[f'cg{round}ratememry']
    sp['imm_recall'] = sp[f'cg{round}dwrdimmrc']
    sp['delayed_recall'] = sp[f'cg{round}dwrddlyrc']

    sp['dementia_class'] = sp.apply(
        dementia_class, round=round, axis=1)

    sp['ad8_score'] = sp.apply(
        ad8_score, round=round, axis=1)

    sp['ad8_binary'] = sp.apply(
        ad8_dementia, round=round, axis=1)

    sp['dementia_class'] = sp.apply(
        update_dementia_class, round=round, axis=1)

    sp['orientation_domain'] = sp.apply(
        orientation_domain, round=round, axis=1)

    sp['clock_test'] = sp.apply(
        clock_test, round=round, axis=1)

    sp['memory_domain'] = sp.apply(
        memory_domain, round=round, axis=1)

    sp = sp.apply(
        domain_binarize, round=round, axis=1)

    sp['dementia_class'] = sp.apply(
        final_update_dementia_class, round=round, axis=1)

    return sp


def round_join(round_list, baseline_vars, longitudinal_vars):

    merged = round_list[0][['spid'] + baseline_vars + longitudinal_vars]
    for i, r in enumerate(round_list[1:]):
        processed_round = r[['spid'] + longitudinal_vars]
        merged = merged.merge(processed_round, on='spid',
                              how='left', suffixes=['', f'_{i+2}'])

    return merged


round_1_cohort = full_cognitive_derivation(round_1_cohort, round=1)
# Year joining

round_2_sp = pd.read_stata('data/NHATS_Round_2_SP_File_v2.dta')
round_3_sp = pd.read_stata('data/NHATS_Round_3_SP_File.dta')
round_4_sp = pd.read_stata('data/NHATS_Round_4_SP_File.dta')
round_5_sp = pd.read_stata('data/NHATS_Round_5B_SP_File.dta')
round_6_sp = pd.read_stata('data/NHATS_Round_6_SP_File_v2.dta')
round_7_sp = pd.read_stata('data/NHATS_Round_7_SP_File.dta')
round_8_sp = pd.read_stata('data/NHATS_Round_8B_SP_File.dta')

round_2_cohort = full_cognitive_derivation(round_2_sp, round=2)
round_3_cohort = full_cognitive_derivation(round_3_sp, round=3)
round_4_cohort = full_cognitive_derivation(round_4_sp, round=4)
round_5_cohort = full_cognitive_derivation(round_5_sp, round=5)
round_6_cohort = full_cognitive_derivation(round_6_sp, round=6)
round_7_cohort = full_cognitive_derivation(round_7_sp, round=7)
round_8_cohort = full_cognitive_derivation(round_8_sp, round=8)

round_list = [round_1_cohort, round_2_cohort, round_3_cohort, round_4_cohort,
              round_5_cohort, round_6_cohort, round_7_cohort, round_8_cohort]

baseline_vars = ['grouping_1_so_status', 'grouping_2_so_status', 'sdoc_sarcopenia', 'sarcopenia', 'obesity', 'high_wc', 'gender', 'race', 'age_category',
                 'smoking_status', 'education', 'heart_disease', 'hypertension', 'arthritis', 'diabetes', 'lung_disease', 'stroke', 'cancer', 'ever_walk', 'ever_walk', 'STEADI_score']

longitudinal_vars = ['self_rated_memory', 'imm_recall', 'delayed_recall', 'clock_test', 'clock_binary',
                     'orientation_domain', 'orientation_binary', 'memory_domain', 'memory_binary', 'ad8_score', 'ad8_binary', 'dementia_class']


test = round_join(round_list, baseline_vars=baseline_vars,
                  longitudinal_vars=longitudinal_vars)
