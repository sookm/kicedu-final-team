run profile1
# f_order
# Out[41]: 

# G2            0.240013
# G1            0.126807
# absences      0.069801
# age           0.037847
# goout         0.032363
# Walc          0.030294
# Mjob          0.030269
# Medu          0.030246
# Fedu          0.029684
# health        0.029659
# freetime      0.028624
# famrel        0.027820
# Fjob          0.026134
# reason        0.025583
# studytime     0.024826
# Dalc          0.020343
# failures      0.018433
# activities    0.016550
# guardian      0.016231
# romantic      0.016180
# traveltime    0.015414
# paid          0.013788
# nursery       0.013654
# famsup        0.012498
# famsize       0.012013
# address       0.011593
# sex           0.011590
# internet      0.010746

# schoolsup     0.009086
# Pstatus       0.007688
# higher        0.004223

# drop columns - 3개, 

student = pd.read_csv('student-mat.csv', engine='python')
student= student.iloc[:,1:] # 첫번째 컬럼이었던 학교 정보 제거 
stu_data = student.iloc[:,:-1]
stu_target = student.iloc[:,-1]

f_drop = ['schoolsup','Pstatus','higher']

stu_data = stu_data.drop(['schoolsup','Pstatus','higher'],axis=1)
stu_target

stu_data['G1'] = pd.cut(stu_data['G1'], 
                     bins=[0,2,5,8,11,14,17,20],
                     include_lowest=True,
                     labels=['1','2','3','4','5','6','7'])

stu_data['G2'] = pd.cut(stu_data['G2'], 
                     bins=[0,2,5,8,11,14,17,20],
                     include_lowest=True,
                     labels=['1','2','3','4','5','6','7'])

stu_target = pd.cut(stu_target, 
                     bins=[0,2,5,8,11,14,17,20],
                     include_lowest=True,
                     labels=['1','2','3','4','5','6','7'])

stu_data.iloc[:,:-2] = f_dummies(stu_data.iloc[:,:-2])

# 2) scaling
m_sc = standard()
m_sc.fit(stu_data)
stu_x_sc = m_sc.transform(stu_data)

# 4. data 분리
train_x, test_x, train_y, test_y = train_test_split(stu_x_sc,
                                                    stu_target,
                                                    random_state=0)

# 1. interaction 적용 data 추출
from sklearn.preprocessing import PolynomialFeatures as poly

m_poly = poly(degree=2)
m_poly.fit(train_x)         # 각 설명변수에 대한 2차항 모델 생성
train_x_poly = m_poly.transform(train_x)   # 각 설명변수에 대한 2차항 모델 생성
test_x_poly  = m_poly.transform(test_x) 

m_poly.get_feature_names()                       # 변경된 설명변수들의 형태 확인
col_poly = m_poly.get_feature_names(stu_data.columns)  # 실제 컬럼이름의 교호작용 출력
 
DataFrame(m_poly.transform(train_x) , 
          columns = m_poly.get_feature_names(stu_data.columns))

# 2. 확장된 데이터셋을 RF에 학습, feature importance 확인
m_rf = rf(random_state=0)
m_rf.fit(train_x_poly, train_y)
m_rf.score(test_x_poly, test_y) # 0.71717
   
m_rf.score(train_x_poly, train_y)


s1 = Series(m_rf.feature_importances_ , index = col_poly)
f_order2 = s1.sort_values(ascending=False) 
f_order2[:30]



# 전진 선택법
l1 = s1.sort_values(ascending=False).index

collist=[]
df_result=DataFrame()

for i in l1 : 
    collist.append(i)
    train_x_sc_poly_sel = DataFrame(train_x_poly, columns = col_poly).loc[:, collist]
    test_x_sc_poly_sel = DataFrame(test_x_poly, columns = col_poly).loc[:, collist]

    m_rf = rf(random_state=0)
    m_rf.fit(train_x_sc_poly_sel, train_y)
    vscore = m_rf.score(test_x_sc_poly_sel, test_y)  
    
    df1 = DataFrame([Series(collist).str.cat(sep='+'), vscore], index=['column_list', 'score']).T
    df_result = pd.concat([df_result, df1], ignore_index=True)
    
df_result.sort_values(by='score', ascending=False)


df_result.sort_values(by='score', ascending=False)[:10]
df_result.sort_values(by='score', ascending=False).iloc[0,0]



#  매개변수 튜닝 
v_score_te = []

for i in range(1,101) :
    m_rf = rf(random_state=0, n_estimators = i)
    m_rf.fit(train_x, train_y)
    v_score_te.append(m_rf.score(test_x, test_y))

import matplotlib.pyplot as plt
plt.plot(np.arange(1,101), v_score_te, color='red')   









#-- 수행2) 전체 컬럼에서 0.00 대 컬럼 제거 ['schoolsup','Pstatus','higher']
f_drop = ['schoolsup','Pstatus','higher']

stu_data1 = stu_data.drop(['schoolsup','Pstatus','higher'],axis=1)
stu_target

stu_data1.iloc[:,:-2] = f_dummies(stu_data1.iloc[:,:-2]) 

# 2. data 변환
# 2) scaling
m_sc = standard()
m_sc.fit(stu_data1)
stu_x_sc = m_sc.transform(stu_data1)

# 4. data 분리
train_x, test_x, train_y, test_y = train_test_split(stu_x_sc,
                                                    stu_target,
                                                    random_state=0)
# 1. interaction 적용 data 추출
from sklearn.preprocessing import PolynomialFeatures as poly

m_poly = poly(degree=2)
m_poly.fit(train_x)         # 각 설명변수에 대한 2차항 모델 생성
train_x_poly = m_poly.transform(train_x)   # 각 설명변수에 대한 2차항 모델 생성
test_x_poly  = m_poly.transform(test_x) 

m_poly.get_feature_names()                       # 변경된 설명변수들의 형태 확인
col_poly = m_poly.get_feature_names(stu_data1.columns)  # 실제 컬럼이름의 교호작용 출력
 
DataFrame(m_poly.transform(train_x) , 
          columns = m_poly.get_feature_names(stu_data1.columns))

# 2. 확장된 데이터셋을 RF에 학습, feature importance 확인
m_rf = rf(random_state=0)
m_rf.fit(train_x_poly, train_y)
m_rf.score(test_x_poly, test_y) # 0.71717
   
s1 = Series(m_rf.feature_importances_ , index = col_poly)
s1.sort_values(ascending=False)[:15]   


# 전체 컬럼 두고/ 성적범주화 / label encoder/scaling / interaction 변수 적용 - 71점 (최고)
#-----------------------------------------------------

f1 = lambda x : 0 if x in ['at_home','other'] else 2
stu_data1['Mjob'] = stu_data1['Mjob'].apply(f1)

f2 = lambda x : 0 if x in ['at_home','other'] else 1
stu['Fjob'] = student['Fjob'].apply(f2)


stu_data1['MFedu'] = stu_data['Medu'] + stu_data['Fedu'] # 0.656565


stu_data1['G2'] = stu_data1['G2'].astype('int')


# -- 부모의 직업 M-heath 2배값, F-teacher 2배값
student = pd.read_csv('student-mat.csv', engine='python')
student= student.iloc[:,1:] # 첫번째 컬럼이었던 학교 정보 제거 
stu_data = studet.iloc[:,:-1]
stu_target = student.iloc[:,-1]

stu_data['G1'] = pd.cut(stu_data['G1'], 
                     bins=[0,2,5,8,11,14,17,20],
                     include_lowest=True,
                     labels=['1','2','3','4','5','6','7'])

stu_data['G2'] = pd.cut(stu_data['G2'], 
                     bins=[0,2,5,8,11,14,17,20],
                     include_lowest=True,
                     labels=['1','2','3','4','5','6','7'])

stu_target = pd.cut(stu_target, 
                     bins=[0,2,5,8,11,14,17,20],
                     include_lowest=True,
                     labels=['1','2','3','4','5','6','7'])

def f_dummies(df) :
    def f1(x) :
        m_label = LabelEncoder()
        m_label.fit(x)
        return m_label.transform(x)
    
    return df.apply(f1, axis=0)

stu_data.iloc[:,:-2] = f_dummies(stu_data.iloc[:,:-2])


student['Mjob']=='health'
m_index = student.Mjob[student['Mjob']=='health'].index

stu_data.Mjob[m_index] = stu_data.Mjob[m_index]*8


student['Fjob']=='teacher'
f_index = student.Fjob[student['Fjob']=='teacher'].index

stu_data.Fjob[f_index] = stu_data.Fjob[f_index]*2


# 2) scaling
m_sc = standard()
m_sc.fit(stu_data)
stu_x_sc = m_sc.transform(stu_data)

# 4. data 분리
train_x, test_x, train_y, test_y = train_test_split(stu_x_sc,
                                                    stu_target,
                                                    random_state=0)

m_rf = rf(random_state=0)
m_rf.fit(train_x, train_y)
   
m_rf.score(test_x, test_y)  #   0.68686

# feature importance 확인
s1 = Series(m_rf.feature_importances_ , index = stu_data.columns)
f_order = s1.sort_values(ascending=False)


# --) G1 * G2
stu_data['G1']
stu_data['G2']


f_mul = lambda x,y : x*y

new_g = list(map(f_mul,stu_data['G1'].astype('int'),stu_data['G2'].astype('int')))
new_g = Series(new_g)

stu_data['G1G2'] = new_g


new_g1 = list(map(f_mul,stu_data['G1'].astype('int'),stu_data['G1'].astype('int')))
new_g1 = Series(new_g1)

stu_data['G1G1'] = new_g1

new_g2 = list(map(f_mul,stu_data['G2'].astype('int'),stu_data['G2'].astype('int')))
new_g2 = Series(new_g2)

stu_data['G2G2'] = new_g2

stu_data = stu_data.drop(['aa'],axis=1)

stu_data.iloc[:,:-5] = f_dummies(stu_data.iloc[:,:-5]) 

stu_data = stu_data.drop(['G2','G1','G1G1'],axis=1)

# 2. data 변환
# 2) scaling
m_sc = standard()
m_sc.fit(stu_data)
stu_x_sc = m_sc.transform(stu_data)

# 4. data 분리
train_x, test_x, train_y, test_y = train_test_split(stu_x_sc,
                                                    stu_target,
                                                    random_state=0)

m_rf = rf(random_state=0)
m_rf.fit(train_x, train_y)
   
m_rf.score(test_x, test_y)  # 0.737 / 0.74747 : 중요도 하위 3개 컬럼 제거시 / # 0.727272 : G2G2, G1G2 만 둘시 

# feature importance 확인
s1 = Series(m_rf.feature_importances_ , index = stu_data.columns)
f_order = s1.sort_values(ascending=False)


# -- 결석수 제곱 :  큰 변화 없음 
stu_data['absences']

new_a = list(map(f_mul,stu_data['absences'].astype('int'),stu_data['absences'].astype('int')))
new_a = Series(new_a)

stu_data['aa'] = new_a

stu_data.iloc[:,:-6] = f_dummies(stu_data.iloc[:,:-6])

# 2) scaling
m_sc = standard()
m_sc.fit(stu_data)
stu_x_sc = m_sc.transform(stu_data)

# 4. data 분리
train_x, test_x, train_y, test_y = train_test_split(stu_x_sc,
                                                    stu_target,
                                                    random_state=0)

m_rf = rf(random_state=0)
m_rf.fit(train_x, train_y)
   
m_rf.score(test_x, test_y)  # 0.737  

# feature importance 확인
s1 = Series(m_rf.feature_importances_ , index = stu_data.columns)
f_order = s1.sort_values(ascending=False)


student.Medu[student.sex=='M'] 


## -----결석수와 성적의 상관관계 확인 

# 결석수와 G2 마지막 성적을 곱한 행을 추가할시 : 0.7777

new_ag = list(map(f_mul,stu_data['absences'].astype('int'),stu_data['G2'].astype('int')))
new_ag = Series(new_ag)

stu_data['AG2'] = new_ag

# Out[16]: 
#     sex  age address famsize  Medu  Fedu  ... absences G1 G2 G1G2  G2G2  AG2

# AG2 : absences 컬럼 값 x G2 

# =>> 중요도가 높은 변수들간의 배수(??) 의 변수 변환 및 추가로 예측점수는 향상시킬 수 있음. 

#-----------------------------------------------------------------------------------------

# --- 성별에 따른 부모의 교육수준 가져오는 행 추가 )
student = pd.read_csv('student-mat.csv', engine='python')
student= student.iloc[:,1:] # 첫번째 컬럼이었던 학교 정보 제거 
stu1 = student.iloc[:,:-1]
stu_target = student.iloc[:,-1]

f_drop = ['schoolsup','Pstatus','higher']

stu1 = stu1.drop(['schoolsup','Pstatus','higher'],axis=1)


stu1['G1'] = pd.cut(stu1['G1'], 
                     bins=[0,2,5,8,11,14,17,20],
                     include_lowest=True,
                     labels=['1','2','3','4','5','6','7'])

stu1['G2'] = pd.cut(stu1['G2'], 
                     bins=[0,2,5,8,11,14,17,20],
                     include_lowest=True,
                     labels=['1','2','3','4','5','6','7'])

stu_target = pd.cut(stu_target, 
                     bins=[0,2,5,8,11,14,17,20],
                     include_lowest=True,
                     labels=['1','2','3','4','5','6','7'])

Pedu=[]
for i in range(0, len(stu1['sex'])) :
    if stu1['sex'][i] == 'F' :
        Pedu.append(stu1['Fedu'][i])
    else :
        Pedu.append(stu1['Medu'][i])

Pedu = Series(Pedu) 
stu1['Pedu'] = Pedu    
    
# ==================================== 2020 09 04 실습문제풀이_0907 참고 ======
 l1=[]
 for i in range(0, len(sub['전체'])) :
     if pd.isnull(sub['전체'][i]) :
         l1.append(sub['전체'][i-1])
     else :
         l1.append(sub['전체'][i])
# =============================================================================

# stu1 = stu1.drop('sex',axis=1)
# stu1 = stu1.drop(f_drop,axis=1)
# stu1 = stu1.drop(['Medu','Fedu'],axis=1)
# stu1 = stu1.drop(['school'],axis=1)
# stu1 = stu1.drop(['G3'],axis=1)

f_mul = lambda x,y : x*y

new_g = list(map(f_mul,stu1['G1'].astype('int'),stu1['G2'].astype('int')))
new_g = Series(new_g)

stu1['G1G2'] = new_g

new_g1 = list(map(f_mul,stu1['G1'].astype('int'),stu1['G1'].astype('int')))
new_g1 = Series(new_g1)

stu1['G1G1'] = new_g1

new_g2 = list(map(f_mul,stu1['G2'].astype('int'),stu1['G2'].astype('int')))
new_g2 = Series(new_g2)

stu1['G2G2'] = new_g2

new_g12 = list(map(f_mul,stu1['G1G2'].astype('int'),stu1['G2'].astype('int')))
new_g12 = Series(new_g12)

stu1['G1G22'] = new_g12


new_ag = list(map(f_mul,stu1['absences'].astype('int'),stu1['G2'].astype('int')))
new_ag = Series(new_ag)

stu1['AG2'] = new_ag


stu1.iloc[:,:-7] = f_dummies(stu1.iloc[:,:-7])

stu1 = stu1.drop(['G2','G1'],axis=1)

# 2) scaling
m_sc = standard()
m_sc.fit(stu1)
stu_x_sc = m_sc.transform(stu1)

# 4. data 분리
train_x, test_x, train_y, test_y = train_test_split(stu_x_sc,
                                                    stu_target,
                                                    random_state=0)

m_rf = rf(random_state=0, n_estimators = 20)
m_rf.fit(train_x, train_y)
   
m_rf.score(test_x, test_y)  # 0.7575  

m_rf.score(train_x, train_y)

# feature importance 확인
s1 = Series(m_rf.feature_importances_ , index = stu_data.columns)
f_order = s1.sort_values(ascending=False)


#  매개변수 튜닝 
v_score_te = []

for i in range(1,101) :
    m_rf = rf(random_state=0, n_estimators = i)
    m_rf.fit(train_x, train_y)
    v_score_te.append(m_rf.score(test_x, test_y))

import matplotlib.pyplot as plt
plt.plot(np.arange(1,101), v_score_te, color='red')   











# -- 성적을 뺀 나머지 컬럼에서 가장 최대변수 찾기 
student = pd.read_csv('student-mat.csv', engine='python')
# student= student.iloc[:,1:] # 첫번째 컬럼이었던 학교 정보 제거 
stu_data = student.iloc[:,:-3]
stu_target = student.iloc[:,-1]

stu_target = pd.cut(stu_target, 
                     bins=[0,2,5,8,11,14,17,20],
                     include_lowest=True,
                     labels=['1','2','3','4','5','6','7'])

stu_data.iloc[:,:-1] = f_dummies(stu_data.iloc[:,:-1])

# 2) scaling
m_sc = standard()
m_sc.fit(stu_data)
stu_x_sc = m_sc.transform(stu_data)

# 4. data 분리
train_x, test_x, train_y, test_y = train_test_split(stu_x_sc,
                                                    stu_target,
                                                    random_state=0)

m_rf = rf(random_state=0)
m_rf.fit(train_x, train_y)
   
m_rf.score(test_x, test_y)  # 0.35353

s1 = Series(m_rf.feature_importances_ , index = stu_data.columns)
f_order = s1.sort_values(ascending=False)





# 결석과 성적의 추이를 선그래프로 따로 표시 하는 그래프 필요 ##### 

# =============================================================================
# # 2) minmax scaling : 집단 내부에서의 최소값을 0 최대값을 1로 특정하여 표준화 시키는 방식. 
# m_sc2 = minmax()
# m_sc2.fit(train_x)       # fitting 하는 순간 각 설명변수 내부의 최소/ 최대값 색출. 
# m_sc2.transform(train_x) # 각 최대/ 최소값에 맞게 scaling 진행 
# m_sc2.transform(test_x)  # 이렇게 하지 않기로 함.(잘못된 방법)
# 
# 
# m_sc2.transform(train_x).min(axis=0) # 각 컬럼(세로방향)의 최소값 구하기.
# m_sc2.transform(train_x).max(axis=0) # 각 컬럼(세로방향)의 최대값 구하기.
# 
# m_sc2 = minmax()
# m_sc2.fit(train_x)                      # 각 설명변수의 최대, 최소 구하기 
# 
# m_sc3 = minmax()
# m_sc3.fit(test_x) 
# =============================================================================


stu_data['absences'].max()

stu_22 = stu_data.loc[:,['absences','G2']]

m_sc22 = minmax()
m_sc22.fit(stu_22)       
stu_x_sc22 = m_sc22.transform(stu_22) 
stu_x_sc2 = DataFrame(stu_x_sc22)
stu_x_sc3 = stu_x_sc2.iloc[:,0].sort_values(ascending=False)



gr_rel['G2'].index

gr_rel = stu_x_sc2
df.columns = ['absences','G2']

# 임의 숫자 index 값으로 하는 매서드 확인 
gr_rel.set_index

stu_x_sc44 = gr_rel.sort_values(by='absences', ascending=False)

df1 = df.index.values
df1 = DataFrame(df1)

df['num'] = df1

df

df3 = df.sort_values(by='absences', ascending=False)
sns.pairplot(df3)


stu_x_sc2.iloc[:,1]

Pedu=[]
for i in range(0, len(stu_x_sc3.index)) :
    if stu_x_sc3[i] == 'F' :
        Pedu.append(stu1['Fedu'][i])
    else :
        Pedu.append(stu1['Medu'][i])

Pedu = Series(Pedu) 
stu1['Pedu'] = Pedu  




s1 = Series(m_rf.feature_importances_ , index = stu1.columns)
f_order = s1.sort_values(ascending=False)


m_sc22 = standard()
m_sc22.fit(stu_22)
stu_x_sc22 = m_sc22.transform(stu_22)
stu_x_sc2 = DataFrame(stu_x_sc22)

df = stu_x_sc2

df.set_index
df

import matplotlib.pyplot as plt
plt.plot(stu_x_sc44.iloc[:,0], label = 'absences')
plt.plot(stu_x_sc44.iloc[:,1], label = 'G2', color = 'red')
plt.legend()

# 한쪽을 정렬을해서 시각화 

d1 = stu_x_sc44.index.values
d1 = DataFrame(d1)
stu_x_sc44['num'] = d1.iloc[:,0]
stu_x_sc44.drop('num',axis=1)


# 선 말고 점으로 시각화 

df3
df4 = np.array(df3)
df4[:,0]

# 시각화 
# 1) figure와 subplot 생성(1X3)
fig, ax = plt.subplots(1,2)

import mglearn 
ax[0].scatter(df4[:,2], df4[:,0], c=mglearn.cm2(0))  # 결석
ax[1].scatter(df4[:,2], df4[:,1], c=mglearn.cm2(1))  # 성적

# 시각화 
# 1) figure와 subplot 생성(1X3)
fig, ax = plt.subplots(1,1)

import mglearn 
ax[0].scatter(df4[:,2], df4[:,0], c=mglearn.cm2(0))  # 결석
ax[0].scatter(df4[:,2], df4[:,1], c=mglearn.cm2(1))  # 성적

# 시각화 
# 1) figure와 subplot 생성(1X3)
fig, ax = plt.subplots(1,2)

import mglearn 
ax[0].scatter(df4[:,0], df4[:,2], c=mglearn.cm2(0))  # 결석
ax[0].scatter(df4[:,1], df4[:,2], c=mglearn.cm2(1))  # 성적





import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt



penguins = sns.load_dataset("penguins")
penguins.head()
fig, axes = plt.subplots(ncols=2, figsize=(8,4))
sns.histplot(data=penguins, x="bill_depth_mm", y="body_mass_g", hue="species", 
             bins=20, ax=axes[0], legend=False)
sns.kdeplot(data=penguins, x="bill_depth_mm", y="body_mass_g", hue="species", 
            fill=True, levels=5, alpha=0.5,
            ax=axes[1], legend=True)

axes[0].set_title("histplot")
axes[1].set_title("kdeplot")

fig.tight_layout()


fig.

sns.kdeplot
sns.kdeplot(df3)

sns.histplot(df3, x="flipper_length_mm", hue="species", multiple="stack")

sns.displot(data=df3, x="flipper_length_mm", hue="species", multiple="stack", kind="kde")

sns.displot(df3)
sns.pairplot(df3)



