
import pandas as pd
from helpers import data_prep

from helpers.eda import check_df
from helpers.eda import grab_col_names
from helpers.data_prep import check_outlier
from helpers.data_prep import replace_with_thresholds
from helpers.data_prep import label_encoder
from helpers.data_prep import rare_encoder
from helpers.data_prep import rare_analyser
from helpers.data_prep import one_hot_encoder
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

def load_dataset():
    data = pd.read_csv("helpers/titanic.csv")
    return data

df = load_dataset()
check_df(df)

def titanic_data_prep(dataframe):

#1.FEATURE ENGINEERING
#değişken isimlerini büyüttüm
    dataframe.columns = [col.upper() for col in dataframe.columns]
#özellik etkileşimleri - Feature interactions/engineering
# Cabin bool
    dataframe["NEW_CABIN_BOOL"] = dataframe["CABIN"].notnull().astype('int')
# Name count
    dataframe["NEW_NAME_COUNT"] = dataframe["NAME"].str.len()
# name word count
    dataframe["NEW_NAME_WORD_COUNT"] = dataframe["NAME"].apply(lambda x: len(str(x).split(" ")))
# name dr
    dataframe["NEW_NAME_DR"] = dataframe["NAME"].apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
    dataframe['NEW_TITLE'] = dataframe.NAME.str.extract(' ([A-Za-z]+)\.', expand=False)
# family size: ailes sayısı
    dataframe["NEW_FAMILY_SIZE"] = dataframe["SIBSP"] + dataframe["PARCH"] + 1
# age_pclass:örn:  yaşı küçük ama 1. sınıf yolcu etkileşimini içerir
    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
# is alone
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) > 0), "NEW_IS_ALONE"] = "NO"
    dataframe.loc[((dataframe['SIBSP'] + dataframe['PARCH']) == 0), "NEW_IS_ALONE"] = "YES"
# age level: yaş aralıklarına kategoriler
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
# sex x age : yaş ve cinsiyet aralıkları
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'
    #değişken sayımız 12'den 22'ye çıktı. 10 tane daha değişken ürettik

#değişkenleri grab_col_names fonksiyonunu ile çağırdım.
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
#Passengerıd'yi numerikten çıkardım. Bana bir faydası yok.
#["AGE", "FARE","NEW_NAME-COUNT","NEW_AGE_PCLASS] numerik değişkenlerim kaldı.
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]

#2. OUTLIERS

#aykırı değer kontrolü(true: aykırı değer var-false)
    for col in num_cols:
        print(col, check_outlier(dataframe, col))
#aykırı değerden kurtulma(fare'de vardı, kurtulduk)
    for col in num_cols:
        replace_with_thresholds(dataframe, col)
#kontrol edelimn
    for col in num_cols:
        print(col, check_outlier(dataframe, col))


#3.MISSING VALUES: eksik değerleri kontrol edelim


#cabin değişkeni üzerinden değişken ürettiğimiz(new_cabin_bool) için cabin değişkenini ucuruyoruz
    dataframe.drop("CABIN", inplace=True, axis=1)
#ticket ve name'i de uçuruyoruz. bu değişkenlerden de değişken üretebiliriz ama mevzu değil şu an.
#new_title'ı name'den türetmiştik zaten. ticket ile de uğraşmak istemediğim için çıkardım
    remove_cols = ["TICKET", "NAME"]
    dataframe.drop(remove_cols, inplace=True, axis=1)
#age ve embarked üzerinden problemler devam ediyor. #missing_values_table(df) ile kontrol edebiliriz bunu.
#age üzerinden türetilmiş değişkenler mevcut.

#yaş değişkenini title'a göre groupby alıp eksiklerini meddyanına göre dolduruyorum.böylelikle yaş gidiyor
    dataframe["AGE"] = dataframe["AGE"].fillna(dataframe.groupby("NEW_TITLE")["AGE"].transform("median"))
    dataframe["NEW_AGE_PCLASS"] = dataframe["AGE"] * dataframe["PCLASS"]
#yaş gitse bile yaşa bağlı değişkenlerde eksiklikler devam ediyor. Bu yüzden yaşa bağlı değişkenleri bir daha oluşturuyoruö
    dataframe.loc[(dataframe['AGE'] < 18), 'NEW_AGE_CAT'] = 'young'
    dataframe.loc[(dataframe['AGE'] >= 18) & (dataframe['AGE'] < 56), 'NEW_AGE_CAT'] = 'mature'
    dataframe.loc[(dataframe['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngmale'
    dataframe.loc[(dataframe['SEX'] == 'male') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturemale'
    dataframe.loc[(dataframe['SEX'] == 'male') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniormale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] <= 21), 'NEW_SEX_CAT'] = 'youngfemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & ((dataframe['AGE'] > 21) & (dataframe['AGE']) <= 50), 'NEW_SEX_CAT'] = 'maturefemale'
    dataframe.loc[(dataframe['SEX'] == 'female') & (dataframe['AGE'] > 50), 'NEW_SEX_CAT'] = 'seniorfemale'

#tekrar kontrol ettiğimizde eksiklerin temizlendiğini görürüz. Mebarked kaldı sadece.
#1'den fazla kategırrik değişken varmış gibi davranıyorum ve embarrked eksiklerini de gideriyorum.
    dataframe = dataframe.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0)
##missing_values_table(df) ile kontrol edebiliriz. eksik kalmadığını göreceğiz

#NOT: ağaç ve gradyan temelli algoritmalarda aykırı ve eksik değerlere duyarsızlık olduğu için ellemek zorunda kalmıyoruz.

#4.LABEL- ENCODING
#kategorik değişkenleri age, ambarked gibi dönüştürmem gerekiyor.
#iki sınıf olanları binary encoding(sex ve new_is_alone iki sınıflıymış.
#iki sınıftan fazla olanları
    binary_cols = [col for col in dataframe.columns if dataframe[col].dtype not in [int, float] and dataframe[col].nunique() == 2]
#binary encoder ları label encoderdan geçiriyoruz.
    for col in binary_cols:
        dataframe = label_encoder(dataframe, col)

#5.RARE-ENCODING
    rare_analyser(df,"SURVIVED", cat_cols)
    dataframe = rare_encoder(df, 0.01, cat_cols)
#rare ile bir toplulaştırma olduğunı gözlemleyebiliriz. uçurulacaklar meydana çıkıyor. 0.000 değerli olanlar.
#sınıf oranı : 0.01.

#6.ONE-HOT
#bütün işlemleri programatik olarak tutmamız lazım. bunu unutmamalıyız.
#değişkenlerin hepsinin sayısalaştırılarak standartlaştırıldığını gözlemleyebiliriz.
    ohe_cols = [col for col in dataframe.columns if 10 >= dataframe[col].nunique() > 2]
    dataframe = one_hot_encoder(dataframe, ohe_cols)
#one-hot-encoding'ten çıkan değişkenlerimin hepsi anlamlı mı? Makine öğrenmesine gitmeden önce bunu iyice incelemeliyim
#yeni dğeişkenlerr türediği için grab_col_names'e yine ihtiyaç duydum
    cat_cols, num_cols, cat_but_car = grab_col_names(dataframe)
    num_cols = [col for col in num_cols if "PASSENGERID" not in col]
#frekans(count) ve oranları(ratio) inceleyerek ssıkıntılı durum olup olmadığını incelemek için rare_analyser'ı çağırdım.
#çöp değişken türedi mi bakmak lazım. new_name_dr biraz düşündürücü
#makine öğrenmesine olabildiğince az yükle gitmemiz lazım
    rare_analyser(df,"SURVIVED", cat_cols)

#NOT: amacımız hayatta kaldı kalmadı durumunu ortaya çıkarak feature üretmek. Bu yüzden oluşturulan değişkenlerin ayırt ediciliği
#ve bilgi taşıyıp taşımaması durumu bunu ifade ediyor. mesela Passangerıd'yi neden çıkardık? çünkü hayatta kkalıp kalmama durumuna
#dair bize bir bilgi taşımıyor. Name değişkeni de öyle. Name'den yararlanıp bilgi taşıyan değişkenler türettiğimiz için
#name değişkenine ihtiyacımız kalmadı. Age değişkeni gibi.

#kullanışsız kolanları getiriyoruz.
    useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < 0.01).any(axis=None)]
# df.drop(useless_cols, axis=1, inplace=True) kullanışsız kolanları istersem silebilirim

#7.STANDART-SCALER
#numerik değişkenleri standartalştırıyorum.
    sc =  StandardScaler()
    dataframe[num_cols] = sc.fit_transform(df[num_cols])

    return dataframe

df_prep = titanic_data_prep(df)
check_df(df_prep)

df_prep.to_pickle("./titanic_data_prep.pickle")

#BUNDAN SONRASI MAKİNE ÖĞRENMESİNE DOĞRU GİDİŞ :)














