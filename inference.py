import cv2

from deploy import build_model

if __name__ == '__main__':
    # visualize_model(model_cfg, deploy_cfg, model,img, device, show_result = True,output_file="./1.jpg")
    
    y_config_path = './config.yaml'
    fcos_model = build_model('fcos3d', y_config_path)
    fcos_model.load_engine()
    image=cv2.imread('/home/cqjtu/Pictures/CAM_FRONT__1526915630862465.jpg')

    fcos_model.detect(fcos_model,image)
    fcos_model.cfx.pop()