import pandas as pd

if __name__=='__main__':
    binary = pd.read_csv('output_binary.csv')
    multiple = pd.read_csv('output.csv')

    all_data = pd.concat([binary, multiple], axis = 1)

    new_label = []
    new_probs = []

    no_probs = [0]*30
    no_probs[0] = 1

    for i in range(len(all_data)):
        if all_data.iloc[i]['binary_label'] == 0:
            new_label.append('no_relation')
            new_probs.append(no_probs)
        else:
            new_label.append(all_data.iloc[i]['pred_label'])
            tmp = eval(all_data.iloc[i,4])
            tmp.insert(0,0)
            new_probs.append(tmp)


    all_df = [new_label, new_probs]
    all_df = pd.DataFrame(all_df).T
    all_df = all_df.reset_index()
    all_df.columns = ['id','pred_label', 'probs']
    all_df.to_csv('output_new.csv')