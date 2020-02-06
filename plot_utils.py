import os
from matplotlib import pyplot as plt

def plot_results_AdvBert(train_losses, val_losses, transfer_results,save_dir, ratio_ids = [0,1,7,8]):
    fig, ((ax1, ax2), (ax3, ax4),(ax5, ax6)) = plt.subplots(nrows=3, ncols=2, figsize=(15,15))

    steps = [int(k) for k in train_losses.keys()]
    steps.sort()

    ax1.set_title('Training Loss')
    ax1.plot(steps, [train_losses[str(s)][0]  for s in steps],'r-',label = 'Sentiment')
    ax1.plot(steps, [train_losses[str(s)][1]  for s in steps],'b-', label = 'Other')
    ax1.plot(steps, [train_losses[str(s)][2]  for s in steps],'r--', label = 'Sentiment_Dis')
    ax1.plot(steps, [train_losses[str(s)][3]  for s in steps],'b--', label = 'Other_Dis')
    ax1.legend()

    steps = [int(k) for k in val_losses.keys()]
    steps.sort()
    ax2.set_title('Validation Loss')
    ax2.plot(steps, [val_losses[str(s)][0]  for s in steps],'r-', label = 'Sentiment')
    ax2.plot(steps, [val_losses[str(s)][1]  for s in steps],'b-', label = 'Other')
    ax2.plot(steps, [val_losses[str(s)][2]  for s in steps],'r--', label = 'Sentiment_Dis')
    ax2.plot(steps, [val_losses[str(s)][3]  for s in steps],'b--', label = 'Other_Dis')
    ax2.legend()

    plt.savefig(os.path.join(save_dir,'all.png'))
    plt.show()
    plt.close()


def plot_results_DEBert( train_losses, val_losses, transfer_results,save_dir,finetune_results = None, model_type = 'TripletBert',eval_ratios = [0.05,0.25,0.5,0.75,0.95], show = True):
    
    val_ratio_num = len(val_losses['0'])
    ratio_ids = list(range(val_ratio_num))
    
    fig = plt.figure(constrained_layout=True, figsize=(25,20))
    gs = fig.add_gridspec(6, len(ratio_ids))

    steps = [int(k) for k in train_losses.keys()]
    steps.sort()

    ax1 = fig.add_subplot(gs[0,:])

    ax1.set_title('Training Loss')
    if model_type == 'TripletBert':
        ax1.plot(steps, [train_losses[str(s)][0]  for s in steps],'r-',label = 'total_loss')
        ax1.plot(steps, [train_losses[str(s)][1]  for s in steps],'b-', label = 'loss_sent')
        ax1.plot(steps, [train_losses[str(s)][2]  for s in steps],'r--', label = 'loss_other')
    elif model_type == 'AdvBert':
        ax1.plot(steps, [train_losses[str(s)][1]  for s in steps],'b-', label = 'loss_sent')
        ax1.plot(steps, [train_losses[str(s)][0]  for s in steps],'r-',label = 'loss_other')
        ax1.plot(steps, [train_losses[str(s)][2]  for s in steps],'b--', label = 'dis_loss_sent')
        ax1.plot(steps, [train_losses[str(s)][3]  for s in steps],'r--', label = 'dis_loss_other')

    ax1.legend()


    
    steps = [int(k) for k in val_losses.keys()]
    steps.sort()

    for i, rat in enumerate(ratio_ids):
        if i == 0:
            ax_0 = fig.add_subplot(gs[1,i])
            ax = ax_0
        else:
            ax = fig.add_subplot(gs[1,i], sharey = ax_0)            
        ax.set_title('val_loss on ratio_id = '+str(eval_ratios[rat]))

        if model_type == 'TripletBert':

            ax.plot(steps, [val_losses[str(s)][i][0]  for s in steps],'r-', label = 'total_loss')
            ax.plot(steps, [val_losses[str(s)][i][1]  for s in steps],'g-', label = 'loss_sent')
            ax.plot(steps, [val_losses[str(s)][i][2]  for s in steps],'b-', label = 'loss_other')
        elif model_type == 'AdvBert':
            ax.plot(steps, [val_losses[str(s)][i][0]  for s in steps],'b-', label = 'loss_sent')
            ax.plot(steps, [val_losses[str(s)][i][1]  for s in steps],'r-',label = 'loss_other')
            ax.plot(steps, [val_losses[str(s)][i][2]  for s in steps],'b--', label = 'dis_loss_sent')
            ax.plot(steps, [val_losses[str(s)][i][3]  for s in steps],'r--', label = 'dis_loss_other')

        ax.legend()

    steps = [int(k) for k in transfer_results.keys()]
    steps.sort()

    ax2 = fig.add_subplot(gs[2,0:])
    ax2.set_title('||W_sent|| / ||W_other|| For Sentiment')
    ax2.axhline(y=1, ls = '--', c = 'k')
    ax2.plot(steps, [transfer_results[str(s)]['0']['combined'][1][0]  for s in steps],'b-', label = 'For Sentiment')
    ax2.legend()


    ax3 = fig.add_subplot(gs[3,0:])
    ax3.set_title('||W_sent|| / ||W_other|| For Genre')
    ax3.axhline(y=1, ls = '--', c = 'k')
    ax3.plot(steps, [transfer_results[str(s)]['0']['combined'][1][1]  for s in steps],'b-', label = 'For Genre')
    ax3.legend()


    for i, rat in enumerate(ratio_ids):
        if i == 0:
            ax_0 = fig.add_subplot(gs[4,i])
            ax = ax_0
        else:
            ax = fig.add_subplot(gs[4,i], sharey = ax_0)
        ax.set_title('Sentiment accuracy  on eval_ratio  = '+str(eval_ratios[rat]))
        ax.plot(steps, [transfer_results[str(s)][str(i)]['sent_enc'][0][0]  for s in steps],'r-', label = 'sent_enc')
        ax.plot(steps, [transfer_results[str(s)][str(i)]['other_enc'][0][0]  for s in steps],'g-', label = 'other_enc')
        ax.legend()

    for i, rat in enumerate(ratio_ids):
        if i == 0:
            ax_0 = fig.add_subplot(gs[5,i])
            ax = ax_0
        else:
            ax = fig.add_subplot(gs[5,i], sharey = ax_0)
        ax.set_title('Genre accuracy on eval_ratio  = '+str(eval_ratios[rat]))
        ax.plot(steps, [transfer_results[str(s)][str(i)]['sent_enc'][0][1]  for s in steps],'r-', label = 'sent_enc')
        ax.plot(steps, [transfer_results[str(s)][str(i)]['other_enc'][0][1] for s in steps],'g-', label = 'other_enc')
        ax.legend()

    # ax4.set_title('Sentiment/Genre accuracy on Source')
    # ax4.plot(steps, [transfer_accs[str(s)]['Sent_orig'][0]  for s in steps], 'r-' , label = 'Sent_Enc For Sentiment')
    # ax4.plot(steps, [transfer_accs[str(s)]['Sent_orig'][1]  for s in steps], 'g-' , label = 'Other_Enc For Sentiment')
    # # ax4.plot(steps, [transfer_accs[str(s)]['Genre'][2]  for s in steps] ,'b-' , label = 'Combined on Target')
    # ax4.plot(steps, [transfer_accs[str(s)]['Genre_orig'][0]  for s in steps], 'r--' , label = 'Sent_Enc For Genre')
    # ax4.plot(steps, [transfer_accs[str(s)]['Genre_orig'][1]  for s in steps], 'g--' , label = 'Other_Enc For Genre')
    # # ax4.plot(steps, [transfer_accs[str(s)]['Genre_orig'][2]  for s in steps], 'b--' , label = 'Combined on Source')
    # ax4.legend()

    # ax5.set_title('Weight ratio for Sentiment/Genre on Target')
    # ind = np.arange(len(steps))    # the x locations for the groups
    # width = 0.35
    # ax5.bar(ind - width/2, [weight_ratios[str(s)]['Sent']  for s in steps],  width, label='For Sentiment', tick_label = steps)
    # ax5.bar(ind + width/2, [weight_ratios[str(s)]['Genre']  for s in steps],  width, label='For Genre', tick_label = steps)
    # ax5.legend()


    # ax6.set_title('Weight ratio for Sentiment/Genre on Source')
    
    # ind = np.arange(len(steps))    # the x locations for the groups
    # width = 0.35
    # ax6.bar(ind - width/2, [weight_ratios[str(s)]['Sent_orig']  for s in steps],  width, label='For Sentiment', tick_label = steps)
    # ax6.bar(ind + width/2, [weight_ratios[str(s)]['Genre_orig']  for s in steps],  width, label='For Genre', tick_label = steps)
    # ax6.legend()

    plt.savefig(os.path.join(save_dir,'all.png'))
    if show:
        plt.show()
    plt.close()

    if finetune_results is not None:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,10))
        steps = list(finetune_results.keys())
        ax1.set_title('Sentiment accuracy  on validation sets with different ratios')
        ratios = [0.05, 0.25, 0.5, 0.75, 0.95]
        for i,r in enumerate(ratios):
            ax1.plot(steps, [finetune_results[str(s)][i][0]  for s in steps], label = 'r = '+str(r))
        ax1.legend()

        ax2.set_title('Genre accuracy  on validation sets with different ratios')
        ratios = [0.05, 0.25, 0.5, 0.75, 0.95]
        for i,r in enumerate(ratios):
            ax2.plot(steps, [finetune_results[str(s)][i][1]  for s in steps], label = 'r = '+str(r))
        ax2.legend()
        plt.savefig(os.path.join(save_dir,'finetune.png'))
        if show:
            plt.show()
        plt.close()



def plot_results_Bert( finetune_results,save_dir,  show = True, pred_both = True):
    steps = list(finetune_results.keys())
    if pred_both:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15,10))
        ax1.set_title('Sentimen`t accuracy  on validation sets with different ratios')
        ratios = [0.05, 0.25, 0.5, 0.75, 0.95]
        for i,r in enumerate(ratios):
            ax1.plot(steps, [finetune_results[str(s)][i][0]  for s in steps], label = 'r = '+str(r))
        ax1.legend()

        ax2.set_title('Genre accuracy  on validation sets with different ratios')
        ratios = [0.05, 0.25, 0.5, 0.75, 0.95]
        for i,r in enumerate(ratios):
            ax2.plot(steps, [finetune_results[str(s)][i][1]  for s in steps], label = 'r = '+str(r))
        ax2.legend()
    else:
        fig, (ax1) = plt.subplots(nrows=1, ncols=1, figsize=(15,10))
        ax1.set_title('Sentiment accuracy  on validation sets with different ratios')
        ratios = [0.05, 0.25, 0.5, 0.75, 0.95]
        for i,r in enumerate(ratios):
            ax1.plot(steps, [finetune_results[str(s)][i]  for s in steps], label = 'r = '+str(r))
        ax1.legend()
    
    plt.savefig(os.path.join(save_dir,'finetune.png'))
    if show:
        plt.show()
    plt.close()
