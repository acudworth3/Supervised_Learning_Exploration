def ab_feature_explore(ab_data):

    label0 = ab_data.all[ab_data.all['room_type'] == 0]
    label1 = ab_data.all[ab_data.all['room_type'] == 1]
    label2 = ab_data.all[ab_data.all['room_type'] == 2]
    assert(len(label0)+len(label1)+len(label2) == len(ab_data.all))

    fig, axes = plt.subplots(1, len(ab_data.features))
    subpl = 0

    for feature in ab_data.features:
        sns.distplot(label0[feature],hist=False,ax=axes[subpl])
        sns.distplot(label1[feature],hist=False,ax=axes[subpl])
        sns.distplot(label2[feature],hist=False,ax=axes[subpl])
        axes[subpl].set(xlabel='# '+str(subpl))

        # axes[subpl].set_title('# '+str(subpl))
        subpl += 1

    fig.tight_layout()
    fig.suptitle('Feature # distribution for each Target')


    plt.show()
    marker = 1



le = preprocessing.LabelEncoder()
ab_corell_data = ab_data.all
ab_corell_data[ab_data.target] = le.fit_transform(ab_corell_data[ab_data.target])
x_vars=list(ab_data.features)
x_vars.append(ab_data.target)

ab_corell_plt_obj = sns.pairplot(ab_corell_data,x_vars=ab_data.features,y_vars=ab_data.target,hue='room_type',diag_kind='hist')
plt.title("this is a test make it a box plot")
ab_corell_plt_obj.savefig('plt_ab_correl.png')
plt.close()
#
# #Pkr
pk_corell_data = pkr_data.all
pk_corell_plt_obj = sns.pairplot(pk_corell_data,x_vars=pkr_data.features,y_vars=pkr_data.target,hue='hand',diag_kind='hist')
plt.title("this is ugly make it a box plot")
pk_corell_plt_obj.savefig('plt_pk_corell')
plt.close()
plt.show()
# le