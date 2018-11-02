import checkpoint as loader
import initializer
import torch

def test_acuracy(dataloaders,checkpoint_name='ic-model.pth',gpu=False):
    # TODO: Do validation on the test set
    cuda=gpu
    model = loader.load_checkpoint(checkpoint_name,cuda)
    correct=0
    total=0
    model.eval()
    if(cuda):
        model.to(device='cuda') 
    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloaders['test']):
            if cuda:
                inputs, labels = inputs.cuda(), labels.cuda()
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Correct'+str(correct))
    print('Total'+str(total))
    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    if (correct/total)>90:
        print ('It was more than 90%')
    else:
        print ('It was less than 90%')


if __name__=="__main__":
     
    image_datasets,dataloaders,dataset_sizes,class_names=initializer.init(root_dir="flowers",stages=['train','valid','test'],train_stage='train')
    test_acuracy(dataloaders,checkpoint_name='ic-model.pth',gpu=False)
