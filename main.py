import torch
from torch.nn import DataParallel

from eval import eval_function
from config import parser
from train import training, training_abnormal
from utlis.utils import select_model


if __name__ == '__main__':
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.multi_label:
        if args.dataset == 'NIH':
            args.num_classes = 8
        elif args.dataset == 'ChesXpert':
            args.num_classes = 5
        else:
            assert "Wrong Dataset"

        print("\n\n Configrations \n Backbone : {} \n Attention used :{} \n Number of classes : {}"
              "\n Global Pooling method :{} \n\n".format(args.backbone, args.attention_map,
                                                         args.num_classes, args.global_pool))
        model = select_model(args)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = DataParallel(model)
        model.to(device)
        training(model, args)
        eval_function(args, model)

    elif not args.multi_label:
        args.num_classes = 1
        print("\n\n Configrations \n Backbone : {} \n Pretrained weights : {} \n Attention used :{}"
              " \n Number of classes : {} \n Global Pooling method :{} \n\n"
              .format(args.backbone, str(args.pretrained), args.attention_map, args.num_classes, args.global_pool))
        model = select_model(args)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = DataParallel(model)
        model.to(device)
        model = training_abnormal(model, args)
        eval_function(args, model)

    # args.pretrained = True
    # print("\n\n Configrations \n Backbone : {} \n Pretrained weights : {} \n Attention used :{}"
    #       " \n Number of classes : {} \n Global Pooling method :{} \n\n"
    #       .format(args.backbone, str(args.pretrained),  args.attention_map, args.num_classes, args.global_pool))
    # model = select_model(args)
    # model = training_abnormal(args)
    # if torch.cuda.device_count() > 1:
    #     print("Let's use", torch.cuda.device_count(), "GPUs!")
    #     model = DataParallel(model)
    # model.to(device)
    # training_PCAM(model, args)
    # model = eval_function(args, model)