
# GENERAL
# group by genre, calculate mean of rank for each
dfx = movies.groupby('Main_Genre')['rank_in_year'].agg('mean')

# group by genre and get values of length for each movie
stats_by_genre = movies.groupby('Main_Genre').apply(lambda x: pd.Series({'length': x['length'].values}))

# group by seasons, count scenes
seasons = lines.groupby('Season').count()['Scene']

# group by seasons, count unique scenes
lines.groupby(["Season", "Episode", "Scene"]).count().reset_index().groupby("Season").count()[["Scene"]]


# create subdataset of the three states
df1 = df[df['State'].isin(['Wisconsin', 'Tennessee', 'Minnesota'])]

# select row n=i 
control_person = control.iloc[i]

#iloc → Position-based indexing (e.g., 0, 1, 2)
#loc → Label-based indexing (e.g., 'row1', 'row2')

trees = np.arange(1, 21) # from 1 to 20 included

# Transform Series to array
(x.Character).array


# DATA VISUALIZATION
import pandas as pd
# 2 variable heatmap
df2 = pd.crosstab(movies['Main_Genre'],movies['studio'])
sns.heatmap(df2, annot=True)

# 3 variable heatmap
df3 = pd.crosstab(movies['Main_Genre'],movies['Genre_2'], values = movies['worldwide_gross'], margins=False, aggfunc='mean')

# 4x4 subplot, length per genre
fig, ax = plt.subplots(4,4,figsize= (8,6), sharey = True, sharex = True)
for i in range(16):
    sbplt = ax[i%4, math.floor(i/4)]
    sbplt.hist(stats_by_genre.iloc[i].values,range = [0,200],bins = 20)
    sbplt.set_title(stats_by_genre.index[i])
    
fig.tight_layout()
fig.text(0.4,0, "Movie length in minutes")
fig.text(0,0.6, "Number of movies", rotation = 90)



# DESCRIBING DATA
# does the data come from a normal distrbution (dist can be also = exp)?
from statsmodels.stats import diagnostic
diagnostic.kstest_normal(df['IncomePerCap'].values, dist = 'norm') # --> if p values < 0.05 : not normal

# if p < alpha --> reject null hypothesis

# bar plot with CIs for income in newyork and california
import seaborn as sns
ax = sns.barplot(x="State", y="IncomePerCap", data=df.loc[df['State'].isin(['New York','California'])])

# bar plot how count of how many items are in each group
import matplotlib.pyplot as plt
groups = ['Treated', 'Control']
counts = [len(treated), len(control)]
plt.bar(groups, counts)

# Pearson coefficient between two variables, amount of linear dependence
from scipy import stats
stats.pearsonr(df['IncomePerCap'],df['SelfEmployed'])

# t-test between two variables
stats.ttest_ind(df.loc[df['State'] == 'New York']['IncomePerCap'], df.loc[df['State'] == 'California']['IncomePerCap'])


# REGRESSION ANALYSIS
# describe death in function of age and creatinine
import statsmodels.formula.api as smf
mod3 = smf.ols(formula='DEATH_EVENT ~ serum_creatinine + age', data=df) #add C() to categorical variables
res3 = mod3.fit()
print(res3.summary())

# OBSERVATIONAL STUDIES
# standardize continuous features
df['age'] = (df['age'] - df['age'].mean())/df['age'].std()

# Propensity score matching
import networkx as nx

def get_similarity(propensity_score1, propensity_score2):
    '''Calculate similarity for instances with given propensity scores'''
    return 1-np.abs(propensity_score1-propensity_score2)

treatment_df = lalonde_data[lalonde_data['treat'] == 1]
control_df = lalonde_data[lalonde_data['treat'] == 0]

# Create an empty undirected graph
G = nx.Graph()

# Loop through all the pairs of instances
for control_id, control_row in control_df.iterrows():
    for treatment_id, treatment_row in treatment_df.iterrows():

        # Calculate the similarity 
        similarity = get_similarity(control_row['Propensity_score'], treatment_row['Propensity_score'])

        # Add an edge between the two instances weighted by the similarity between them
        G.add_weighted_edges_from([(control_id, treatment_id, similarity)])

# Generate and return the maximum weight matching on the generated graph
matching = nx.max_weight_matching(G)

matched = [i[0] for i in list(matching)] + [i[1] for i in list(matching)]
balanced_df_1 = lalonde_data.iloc[matched]



# SUPERVISED LEARNING 
# fill NaNs with mean
X = X.fillna(X.mean())

# cross validation, returns 'scoring' metrics
scores = cross_validate(model, X, y, cv=10, scoring=scoring)


# 8. UNSUPERVISED LEARNING / CLUSTERING
# This create some artifical clusters with standard dev. = 2
X, _, centers = make_blobs(n_samples=total_samples, 
                           centers=num_clusters, 
                           cluster_std=2,
                           n_features=2, #n feature is the dimension of the data
                           return_centers=True, 
                           random_state=42)

plt.scatter(X[:,0], X[:,1], alpha=0.6)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Artificial clusters (%s samples)" % total_samples)

for c in centers:
    plt.scatter(c[0], c[1], marker="+", color="red")

# Cluster the data with the current number of clusters (KMeans)
kmean = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
# Plot the data by using the labels as color
ax.scatter(X[:,0], X[:,1], c=kmean.labels_, alpha=0.6)
ax.set_title("%s clusters"%n_clusters)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
# Plot the centroids
for c in kmean.cluster_centers_:
    ax.scatter(c[0], c[1], marker="+", color="red")


# Find best number of clusters
# 1. silhouette score (find max)
silhouettes = []

for k in range(2, 11):
    # Cluster the data and assigne the labels
    labels = KMeans(n_clusters=k, random_state=10).fit_predict(X)
    # Get the Silhouette score
    score = silhouette_score(X, labels)
    silhouettes.append({"k": k, "score": score})
    
silhouettes = pd.DataFrame(silhouettes)

plt.plot(silhouettes.k, silhouettes.score)
plt.xlabel("K")
plt.ylabel("Silhouette score")

#2. Elbow method : Find the "elbow" in the curve of the Sum of Squared Errors
def plot_sse(features_X, start=2, end=11):
    sse = []
    for k in range(start, end):
        # Assign the labels to the clusters
        kmeans = KMeans(n_clusters=k, random_state=10).fit(features_X)
        sse.append({"k": k, "sse": kmeans.inertia_})

    sse = pd.DataFrame(sse)
    # Plot the data
    plt.plot(sse.k, sse.sse)
    plt.xlabel("K")
    plt.ylabel("Sum of Squared Errors")
    
plot_sse(X)

# Visualize high dimension data
# 1. t-SNE
X_reduced_tsne = TSNE(n_components=2, init='random', learning_rate='auto', random_state=0).fit_transform(X10d)
print("The features of the first sample are: %s" % X_reduced_tsne[0])

# 2. PCA
X_reduced_pca = PCA(n_components=2).fit(X10d).transform(X10d)
print("The features of the first sample are: %s" % X_reduced_pca[0])

# 3. Visualize results
fig, axs = plt.subplots(1, 2, figsize=(7,3), sharey=True)

# Cluster the data in 3 groups
labels = KMeans(n_clusters=3, random_state=0).fit_predict(X10d)

# Plot the data reduced in 2d space with t-SNE
axs[0].scatter(X_reduced_tsne[:,0], X_reduced_tsne[:,1], c=labels, alpha=0.6)
axs[0].set_title("t-SNE")

# Plot the data reduced in 2d space with PCA
axs[1].scatter(X_reduced_pca[:,0], X_reduced_pca[:,1], c=labels, alpha=0.6)
axs[1].set_title("PCA")

# DBSCAN (low eps: more clusters, high eps: less clusters)
X_moons, _ = make_moons(500, noise=0.05, random_state=0)
labels = DBSCAN(eps=eps).fit_predict(X_moons)
ax.scatter(X_moons[:,0], X_moons[:,1], c=labels, alpha=0.6)


# 9. HANDLING TEXT DATA
