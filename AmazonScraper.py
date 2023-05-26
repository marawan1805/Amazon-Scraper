import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import cchardet
from sklearn.cluster import KMeans


#Scraping mobile phones info from Amazon and performing EDA and clustering


#we extract the webpage we're interested in using beautiful soup
def extract(page):
    requests_session = requests.Session()
    url = f'https://www.amazon.ae/s?i=electronics&rh=n%3A15415001031&fs=true&page={page}&qid=1658342605&ref=sr_pg_2'
    r = requests_session.get(url)
    soup = BeautifulSoup(r.content, 'lxml')
    return soup

#we create a product dictionary using the class headers in the product card
def transform(soup):
    divs = soup.find_all('div', class_= 's-card-container')
    for item in divs:
        title = item.find('span', class_='a-size-base-plus').text.strip()
        try:
            price = item.find('span', class_='a-price-whole').text.strip().replace(',','') + item.find('span', class_='a-price-fraction').text.strip()
        except:
            price = ''
        try:
            rating_out_of_five = item.find('span', class_='a-icon-alt').text[0:3]
        except:
            rating_out_of_five = ''
        try:
            number_of_reviews = item.find('span', class_='a-size-base s-underline-text').text.replace(',','')
        except:
            number_of_reviews = ''
             
        product = {
                'Title': title[0:40],
                'Rating': rating_out_of_five,
                'Price': price,
                'Reviews': number_of_reviews
        }
        productList.append(product)
    return

productList = []

#we repeat the process for the first 30 pages (or as desired)
for i in range(1,31,1):
    print(f'Getting page, {i}')
    c = extract(i)
    transform(c)

#we convert our product list into a dataframe
df = pd.DataFrame(productList)
#df.to_csv(r"path\products.csv")

#EDA
print('\n')
print(df.head())
print('\n')
df['Rating'] = pd.to_numeric(df['Rating'],errors='coerce')
df['Price'] = pd.to_numeric(df['Price'],errors='coerce')
df = df[~df['Reviews'].isin(['More', 'Only', ''])]
df = df[~df['Price'].isin([''])]
df['Reviews'] = df['Reviews'].astype(int)
df.info()
print('\n')
print(df.describe())

#Interested in products with number of reviews > 200
df1 = df[(df['Reviews'] >= 200)]
df1 = df1.dropna()
print('\n')
print(df1.head())
# we notice that almost all products with a number of 
# reviews > 200 have a rating > 4

print('\n')
print(df1.isnull().sum())
#df1.to_csv(r"path\products.csv")

#number of items in every price range
plt.hist(df['Price'], bins = 10, ec='black')
plt.xlabel('Price')
plt.ylabel('Number of items')
plt.show()

#scatter plot
x = df1['Rating']
y = df1['Price']
plt.scatter(x, y, label= "stars", color= "m", 
            marker= "*", s=30)
plt.xlabel('Rating out of 5')
plt.ylabel('Price in AED')
plt.show()

#clustering
X = df1.iloc[:,1:3].values

#we use elbow method to determine optimal K value
wcss = [] 
for i in range(1, 11): 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X) 
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS') 
plt.show()

#elbow point is 5, so we train our data with a number of clusters = 5
kmeans = KMeans(n_clusters = 5, init = "k-means++", random_state = 42)
y_kmeans = kmeans.fit_predict(X)
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 60, c = 'red', label = 'Cluster1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 60, c = 'blue', label = 'Cluster2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 60, c = 'green', label = 'Cluster3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 60, c = 'violet', label = 'Cluster4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 60, c = 'yellow', label = 'Cluster5') 
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroids')
plt.xlabel('Rating')
plt.ylabel('Price')
plt.legend() 
plt.show()