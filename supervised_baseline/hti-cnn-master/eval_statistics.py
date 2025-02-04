import numpy as np
import pandas


def main():

    class_aucs3 = np.load('/SSL_data/supervisedCNNs/Bray_bioactives/supervised_outputs/results/class_aucs3.npy')
    class_f1s3 = np.load('/SSL_data/supervisedCNNs/Bray_bioactives/supervised_outputs/results/class_f1s3.npy')

    class_aucs2 = np.load('/SSL_data/supervisedCNNs/Bray_bioactives/supervised_outputs/results/class_aucs2.npy')
    class_f1s2 =np.load('/SSL_data/supervisedCNNs/Bray_bioactives/supervised_outputs/results/class_f1s2.npy')

    class_aucs1 = np.load('/SSL_data/supervisedCNNs/Bray_bioactives/supervised_outputs/results/class_aucs1.npy')
    class_f1s1 = np.load('/SSL_data/supervisedCNNs/Bray_bioactives/supervised_outputs/results/class_f1s1.npy')

    class_aucs_avg = (class_aucs1 + class_aucs2 + class_aucs3)/3
    class_f1s_avg =  (class_f1s1 + class_f1s2 + class_f1s3)/3

    print('class_aucs tabel values')
    for index, val in np.ndenumerate(class_aucs_avg):
        print (val)

    print('------------------------------')
    
    print('class_f1 tabel values')
    for index, val in np.ndenumerate(class_f1s_avg):
        print (val)

    mean_auc = float(np.mean(class_aucs_avg))
    mean_f1 = float(np.mean(class_f1s_avg))

    over9 = (class_aucs_avg >= 0.9).sum()
    over8 = (class_aucs_avg >= 0.8).sum()
    over7 = (class_aucs_avg >= 0.7).sum()

    # write statistics

    print(' * AUC {auc:.3f}\tAUC>0.9 {auc9:.3f}\tAUC>0.8 {auc8:.3f}\tAUC>0.7 {auc7:.3f}\tF1 {f1:.3f}'.format( auc=mean_auc, auc9=over9, auc8=over8, auc7=over7, f1=mean_f1))

    print("Done")




if __name__ == '__main__':
    main()
