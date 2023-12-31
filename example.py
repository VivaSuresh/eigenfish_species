from pathlib import Path

import cv2

import eigenfish

import eigenfish
import util

if __name__ == "__main__":
    print("Loading training images")
    fish_image_path = Path("data/fish")
    fish_imgs = list(fish_image_path.glob("*"))

    rgb_mat, shape = util.load_img_mat(fish_imgs)
    
    print("Training model")
    ef = eigenfish.Eigenfish(shape)

    ef.train_rgb(rgb_mat, ["fish" for i in range(0, 21)])

    # print("Loading test images")
    # test_imgs = (["example_data/fish/%d.jpg" % i for i in range(15, 20)] +
    #              ["example_data/nofish/%d.jpg" % i for i in range(15, 20)])
    # test_mat = util.load_img_mat(test_imgs)[0]

    # print("Classifying test images")
    # labels = ef.classify(test_mat)
    # print("Labels: {}".format(labels))

    # print("Cross-validating test images")
    # pct = ef.cross_validate(test_mat, (["fish" for i in range(5)] +
    #                                    ["no fish" for i in range(5)]))
    # print("Percent correct: {}%".format(pct * 100))

    # print("Saving trained model")
    # ef.save("example.pkl")

    # print("Reloading saved model")
    # del ef
    # ef = eigenfish.Eigenfish(shape)
    # ef.load("example.pkl")
