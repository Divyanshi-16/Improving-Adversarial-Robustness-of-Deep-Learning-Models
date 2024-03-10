# Improving-Adversarial-Robustness-of-Deep-Learning-Models
This implementation is inspired by the CVPR paper, "Boosting Accuracy and Robustness of Student Models via Adaptive Adversarial Distillation".</br>
### Some Important Terms: -</br>
1) Teacher-Student Architectures: These have been considered as a means of computational-effective and high-performing deployment of DNNs ( lightweight version of the models) in mobile devices with limited resources for prompt inference results.</br>
2) Adversarial Distillation(AD): It can be formulated as a min-max optimization problem. It aims to enable the student model to inherit not only the prediction accuracy but also the adversarial robustness from a robust teacher model under the paradigm of robust optimization.</br>
3) Min-Max Optimization Problem: The inner maximization problem seeks to find the worst-case perturbation that maximizes the loss, while the outer minimization problem optimizes the model's parameters to minimize the loss over the entire dataset.</br>
4) Knowledge Distillation: It aims at distilling knowledge of larger teacher models into small student models and is widely adopted in model compression. In the ordinary knowledge distillation, the student model is expected to inherit clean accuracy from the teacher model without consideration on adversarial robustness.
5) Adaptive Adversarial Distillation (AdaAD): (proposed by the paper) It fully involves a robust teacher model to adaptively search for more representative inner results in the knowledge distillation process. It maximizes the prediction discrepancy between teacher and student models in the min-max framework. It adaptively searches for optimal match points in the inner optimization. This enables a much larger search radius (perturbation-limit) in local neighbourhoods, which significantly enhances the robustness of student models.</br>
We first use gradient descent algorithm to adaptively search the upper bound of the prediction discrepency between the student and teacher model in the inner optimization. </br>
Then we minimize the upper bound in outer optimization to perform distillation.</br>
6) Adaptive Introspective Adversarial Distillation (AdaIAD): It doesn't conduct AD for those points on which teacher models make wrong predictions.
### Experimental Setup: -</br>
- Dataset: CIFAR-10
- Teacher Model: WideResNet-34-10
- Student Model: ResNet-18
- Methods Implemented: AdaAD, AdaIAD
- SGD momentum optimizer with an initial learning rate 0.1, momentum 0.9, and weight decay 5e-4.
- 200 training epochs, the learning rate is divided by 10 at the 100th and 150th epochs.
-  The number of iterations during the inner optimization is set to 10 with step size 2/255, and the total perturbation bound is 8/255 under L∞ constrain.
-  Hyper-parameter α is set to 1.0
-  Standard distillation temperature
-  random cropping and flipping for data augmentation during the whole training process.
-  Pytorch framework.
### Evaluation Metrics: -</br>
We use natural/clean accuracy on natural test samples and robust accuracy on adversarial test samples to demonstrate model performance. We consider 4 representative adversarial attacks including FGSM, PGD, CW2 (constrained by l2 norm), and AutoAttack (AA). For FGSM, PGD, and AA, the maximum perturbation size is set to 8/255, while PGD adopts 10 steps with step size 2/255. The balance constant in CW is set to 0.1.
