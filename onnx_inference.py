import onnx
import onnxruntime as ort
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
# 분류다 평가지표 분류 평가지표
from onnx_to_nengo_model import toNengoModel, classification_accuracy, classification_error, objective
import re
import nengo
import nengo_dl
import tensorflow as tf

""" 
onnx 파일이 
skl2onnx로 -> https://github.com/onnx/onnx/blob/master/docs/Operators-ml.md
keras2onnx로 -> https://github.com/onnx/onnx/blob/master/docs/Operators.md
tensorflow2onnx 로 만들어 졌을 때를 가정
"""

# 모델 판별 Module 클래스
# 주요기능 1. 해당 모델이 어떤 프레임워크로 만들어졌는지
# 주요기능 2. 해당 모델이 어떤 연산자 기본인지 -> ai.onnx / onnx.ml /
class Distinguish_onnx:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_framework = None
        self.testX = None; self.testY = None; # test data
        self.onnx_load_model = onnx.load_model(model_path)
        self.model_type = None

    # model type 이 snn 이지 아닌지 구분
    # 어떠한 프레임워크로 만들어 졌는가.
    def getModelFramework(self):
        # onnx 만든 프레임워크
        # model type 이 snn 이지 아닌지 구분
        if self.model_type == None:
            self.model_type = self.onnx_load_model.producer_name
            self.model_type = self.model_type.replace('2onnx','')
            print(self.model_type)
            self.model_framework = self.model_type
            if self.model_type == 'skl':
                self.model_framework = 'scikit-learn'
                return 'ML ' + self.model_framework
            elif self.model_type == 'keras':
                self.model_framework = 'keras'
                return 'DL ' + self.model_framework
            elif self.model_type =='tf':
                self.model_framework = 'tensorflow'
                return 'DL ' + self.model_framework
        elif self.model_type == 'snn':
            print('model_type 이 snn 입니다')
            self.model_framework = 'nengo'
            return 'SNN' + self.model_framework

        else: # 오류 발생시
            raise ValueError('사용되어진 Framework를 알수 없습니다')

    # 어떠한 연산자를 기본으로 사용하고 있는가
    def getModelOperator(self):
        for i in range(len(self.onnx_load_model.graph.node)):
            op_type = self.onnx_load_model.graph.node[i].op_type.lower()
            if op_type == "lif" or op_type == "lifrate" or op_type == "adaptivelif" \
                    or op_type == "adaptivelifrate" or op_type == "izhikevich" \
                    or op_type == "softlifrate":
                self.model_type = 'snn'
                print(self.model_type)
                return
        return

    #dl과 ml의 구분_"ai.onnx"
    def getModeldomain(self):
        model_domain_opertor = self.onnx_load_model.domain
        return model_domain_opertor

    # ONNX Runtime 으로 추론 - ml, dl
    def ort_run(self, testX, testY):
        self.testX = np.load(testX);
        self.testY = np.load(testY)

        sess = ort.InferenceSession(model_path)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name

        print(type(self.testX), type(self.testY))
        pred_onx = sess.run([label_name], {input_name: self.testX.astype(np.float32)})
        print('-- 추론 Complete -- ')

        # 테스트 predict 결과들 비교 (평가지표 보기위함)
        pred = np.round(np.array(pred_onx).flatten().tolist())

        test = np.array(np.array(self.testY).flatten().tolist())
#        print(pred, type(pred), pred.shape)
#        print(test, type(test), test.shape)
        k_accuracy = float(accuracy_score(test, pred))
        print('accuracy : ' , k_accuracy)

    def nengo_run(self, testX, testY):
        print('mnist data 준비')
        (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
        print(train_images.shape, test_images.shape, train_labels.shape, test_labels.shape)

        # flatten images
        train_images = train_images.reshape((train_images.shape[0], -1))
        test_images = test_images.reshape((test_images.shape[0], -1))

        # to nengo 로 한거지
        otn = toNengoModel("model/lenet-1_snn.onnx")
        model = otn.get_model()
        inp = otn.get_inputProbe()
        pre_layer = otn.get_endLayer()

        # 돌리는 것
        with model:
            out_p = nengo.Probe(pre_layer)
            out_p_filt = nengo.Probe(pre_layer, synapse=0.01)

        minibatch_size = 2000

        # ----------------------------------------------------------- run
        sim = nengo_dl.Simulator(model, minibatch_size=minibatch_size, device="/cpu:0")

        # add single timestep to training data
        train_images = train_images[:, None, :]
        train_labels = train_labels[:, None, None]

        # when testing our network with spiking neurons we will need to run it
        # over time, so we repeat the input/target data for a number of
        # timesteps.
        n_steps = 30
        test_images = np.tile(test_images[:, None, :], (1, n_steps, 1))
        test_labels = np.tile(test_labels[:, None, None], (1, n_steps, 1))

        print('-- train data, test data 준비 완료')
        print('-- evaluate start')

        # note that we use `out_p_filt` when testing (to reduce the spike noise)
        sim.compile(loss={out_p_filt: classification_accuracy})

        print("Accuracy before training:", sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)["loss"], )

        do_training = True
        if do_training:
            # run training
            sim.compile(

                optimizer=tf.optimizers.RMSprop(0.001),
                loss={out_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True)},
            )
            sim.fit(train_images, {out_p: train_labels}, epochs=1)

            # save the parameters to file
            sim.save_params("mnist_params")
            print('save_params')
        # else:
        #     # download pretrained weights
        #     urlretrieve(
        #         "https://drive.google.com/uc?export=download&"
        #         "id=1l5aivQljFoXzPP5JVccdFXbOYRv3BCJR",
        #         "mnist_params.npz",
        #     )
        #
        # # load parameters
        # print('load_params')
        # sim.load_params("./mnist_params")

        sim.compile(loss={out_p_filt: classification_accuracy})
        print("Accuracy after training:", sim.evaluate(test_images, {out_p_filt: test_labels}, verbose=0)['loss'], )
        data = sim.predict(test_images[:minibatch_size])
        sim.close()
        print('simulator 종료')

# main 코드
if __name__ == "__main__":
    print('++ 다음중 원하는 모델 이름을 입력하세요') # ==> 해당 model 파일 list 출력
    print('---모델 리스트 : rf_iris / lenet-1 / lenet-1_snn')
    model_name = input()
    # model_name = 'rf_iris' #'rf_iris'
    # model_name = 'lenet-1_snn'
    model_path = 'model/' + model_name + '.onnx'

    # 모델 판별 Class 로드
    # 0. 생성자
    distinguish_onnx = Distinguish_onnx(model_path) # 모델 경로
    print('--- 사용되어진 Model Name은 ', model_name +'.onnx 입니다')

    # snn 인지 아닌지 먼저 판별
    # 1. op_type 확인 - snn 활성화 함수 (lif, .. 가 들어있으면 snn 으로 정함)
    distinguish_onnx.getModelOperator()

    # 2. 사용된 딥러닝 Framework
    model_framework = distinguish_onnx.getModelFramework()
    print('--- 사용되어진 Model Framework는 ', model_framework, '입니다')

    # 3. 사용되어진 도메인 연산자는 어떤건지.
    # model_operator = distinguish_onnx.getModelOperator()
    # print('--- 사용되어진 모델 도메인 Operators는 ', model_operator, 'operator 입니다')

    print('++ Test File(.npy)를 입력해주세요')
    print('--- 선택모델이 ML 인 경우 iris_라고 입력, DL, SNN 인경우 mnist_로 입력')
    test_file_name = input()
    #test_file_name = 'iris_'
    #test_file_name = 'mnist_'
    testX = 'test_data/' + test_file_name + 'X_test.npy'
    testY = 'test_data/' + test_file_name + 'Y_test.npy'
    # 3. ONNX Runtime 추론
    if model_framework =='SNNnengo':
        distinguish_onnx.nengo_run(testX, testY)
    else:
        model_domain =distinguish_onnx.getModeldomain()
        if model_domain == 'ai.onnx':
            # ml의 경우
            # print("ml입니다")
            distinguish_onnx.ort_run(testX, testY)
        else: # onnx 일경우
            # dl의 경우
            # print("dl입니다")
            distinguish_onnx.ort_run(testX, testY)

#sci_model_path = 'model/' + model_name
#dl_model_path =  'model/' + model_name