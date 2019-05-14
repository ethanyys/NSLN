import tensorflow as tf
import numpy as np

tf.set_random_seed(1337)
DUMMY_VAL = -1000

class CRF(object):
    # def __init__(self, batch_size, num_classes, num_steps, tensor, targets, targets_transition, length):
    def __init__(self, batch_size, num_classes, num_steps, tensor):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_steps = num_steps
        
        self.logits = self._linear_trans(tensor)
        
        self.tags_scores = tf.reshape(self.logits, [self.batch_size, self.num_steps, self.num_classes])
        self.transitions = tf.get_variable("transitions", [self.num_classes + 1, self.num_classes + 1])
        # dummy_val = -1000
        class_pad = tf.Variable(DUMMY_VAL * np.ones((self.batch_size, self.num_steps, 1)), dtype=tf.float32)
        self.observations = tf.concat([self.tags_scores, class_pad], 2)
            
        begin_vec = tf.Variable(np.array([[DUMMY_VAL] * self.num_classes + [0] for _ in range(self.batch_size)]), trainable=False, dtype=tf.float32)
        end_vec = tf.Variable(np.array([[0] + [DUMMY_VAL] * self.num_classes for _ in range(self.batch_size)]), trainable=False, dtype=tf.float32) 
        begin_vec = tf.reshape(begin_vec, [self.batch_size, 1, self.num_classes + 1])
        end_vec = tf.reshape(end_vec, [self.batch_size, 1, self.num_classes + 1])
            
        self.observations = tf.concat([begin_vec, self.observations, end_vec], 1)
            
    def _linear_trans(self, tensor):
        with tf.variable_scope("non-linear-trans"):
            non_linear_w = tf.get_variable("non_linear_w", [tensor.get_shape().as_list()[-1], self.num_classes])
            non_linear_b = tf.get_variable("non_linear_b", [self.num_classes])
            logits = tf.matmul(tensor, non_linear_w) + non_linear_b
        return logits
    
    
    '''
        full annotation learning
    '''
    def _calculate_target_path_score(self, targets, targets_transition):
        self.mask = tf.cast(tf.reshape(tf.sign(targets),[self.batch_size * self.num_steps]), tf.float32)
            
        # point score
        self.point_score = tf.gather(tf.reshape(self.tags_scores, [-1]), tf.range(0, self.batch_size * self.num_steps) * self.num_classes + tf.reshape(targets,[self.batch_size * self.num_steps]))
        self.point_score *= self.mask

        # transition score
        self.trans_score = tf.gather(tf.reshape(self.transitions, [-1]), targets_transition)
        
        # real score
        self.target_path_score = tf.reduce_sum(self.point_score) + tf.reduce_sum(self.trans_score)  
    
    
    def logsumexp(self, x, axis=None):
        x_max = tf.reduce_max(x, reduction_indices=axis, keepdims=True)
        x_max_ = tf.reduce_max(x, reduction_indices=axis)
        return x_max_ + tf.log(tf.reduce_sum(tf.exp(x - x_max), reduction_indices=axis))

    
    def forward(self, observations, transitions, length, is_viterbi=True, return_best_seq=True):
        length = tf.reshape(length, [self.batch_size])
        transitions = tf.reshape(tf.concat([transitions] * self.batch_size, 0), [self.batch_size, self.num_classes+1, self.num_classes+1])
        observations = tf.reshape(observations, [self.batch_size, self.num_steps + 2, self.num_classes+1, 1])
        observations = tf.transpose(observations, [1, 0, 2, 3])
        
        previous = observations[0, :, :, :]
        max_scores = []
        max_scores_pre = []
        alphas = [previous]
        
        previous_v = observations[0, :, :, :]

        for t in range(1, self.num_steps + 2):
            previous = tf.reshape(previous, [self.batch_size, self.num_classes+1, 1])
            current = tf.reshape(observations[t, :, :, :], [self.batch_size, 1, self.num_classes+1])
            alpha_t = previous + current + transitions
            if is_viterbi:
                previous_v = tf.reshape(previous, [self.batch_size, self.num_classes+1, 1])
                previous_t = tf.reduce_max(previous_v + current + transitions, reduction_indices=1)
                previous_pre_t = tf.argmax(previous_v + current + transitions, axis=1)
                max_scores.append(previous_t)
                max_scores_pre.append(previous_pre_t)
                previous_v = previous_t
            alpha_t = tf.reshape(self.logsumexp(alpha_t, axis=1), [self.batch_size, self.num_classes+1, 1])
            alphas.append(alpha_t)
            previous = alpha_t
            

        alphas = tf.reshape(tf.concat(alphas, 0), [self.num_steps + 2, self.batch_size, self.num_classes+1, 1])
        alphas = tf.transpose(alphas, [1, 0, 2, 3])
        alphas = tf.reshape(alphas, [self.batch_size * (self.num_steps + 2), self.num_classes+1, 1])

        last_alphas = tf.gather(alphas, tf.range(0, self.batch_size) * (self.num_steps + 2) + length)
        last_alphas = tf.reshape(last_alphas, [self.batch_size, self.num_classes+1, 1])

        max_scores = tf.reshape(tf.concat(max_scores, 0), (self.num_steps + 1, self.batch_size, self.num_classes+1))
        max_scores_pre = tf.reshape(tf.concat(max_scores_pre, 0), (self.num_steps + 1, self.batch_size, self.num_classes+1))
        max_scores = tf.transpose(max_scores, [1, 0, 2])
        max_scores_pre = tf.transpose(max_scores_pre, [1, 0, 2])

        return tf.reduce_sum(self.logsumexp(last_alphas, axis=1)), max_scores, max_scores_pre
    
    '''
        partial annotation learning
    '''
    def logsumexp_PA(self, x, pre_pa_path, axis=None):
        x_max = tf.reduce_max(x, reduction_indices=axis, keepdims=True)
        x_max_ = tf.reduce_max(x, reduction_indices=axis)
        # pre_pa_path = tf.reshape(pre_pa_path, [self.batch_size, self.num_classes+1, 1])
        return x_max_ + tf.log(tf.reduce_sum(tf.multiply(pre_pa_path, tf.exp(x - x_max)), reduction_indices=axis))
    
    
    def PA_forward(self, observations, transitions, length, y_batch, is_viterbi=True):
        length = tf.reshape(length, [self.batch_size])
        transitions = tf.reshape(tf.concat([transitions] * self.batch_size, 0), [self.batch_size, self.num_classes+1, self.num_classes+1])
        observations = tf.reshape(observations, [self.batch_size, self.num_steps + 2, self.num_classes+1, 1])
        observations = tf.transpose(observations, [1, 0, 2, 3])
        
        y_batch = tf.reshape(y_batch, [self.batch_size, self.num_steps + 2, self.num_classes+1, 1])
        last_y_batch = tf.reshape(y_batch, [self.batch_size * (self.num_steps + 2), self.num_classes+1, 1])
        last_y_batch = tf.gather(last_y_batch, tf.range(0, self.batch_size) * (self.num_steps + 2) + length)
        y_batch = tf.transpose(y_batch, [1, 0, 2, 3])
        
        previous = observations[0, :, :, :]
        max_scores = []
        max_scores_pre = []
        alphas = [previous]
        
        previous_v = observations[0, :, :, :]
        
        # pa
        previous_pa = observations[0, :, :, :]
        alphas_pa = [previous_pa] 

        for t in range(1, self.num_steps + 2):
            previous = tf.reshape(previous, [self.batch_size, self.num_classes+1, 1])
            current = tf.reshape(observations[t, :, :, :], [self.batch_size, 1, self.num_classes+1])
            alpha_t = previous + current + transitions
            if is_viterbi:
                # max_scores.append(tf.reduce_max(alpha_t, reduction_indices=1))
                # max_scores_pre.append(tf.argmax(alpha_t, axis=1))
                previous_v = tf.reshape(previous, [self.batch_size, self.num_classes+1, 1])
                previous_t = tf.reduce_max(previous_v + current + transitions, reduction_indices=1)
                previous_pre_t = tf.argmax(previous_v + current + transitions, axis=1)
                max_scores.append(previous_t)
                max_scores_pre.append(previous_pre_t)
                previous_v = previous_t
            alpha_t = tf.reshape(self.logsumexp(alpha_t, axis=1), [self.batch_size, self.num_classes+1, 1])
            alphas.append(alpha_t)
            previous = alpha_t
            
            ### pa add operation
            previous_pa = tf.reshape(previous_pa, [self.batch_size, self.num_classes+1, 1])
            alpha_t_pa = previous_pa + current + transitions
            pre_pa_path = y_batch[t-1, :, :, :]
            
            alpha_t_pa = tf.reshape(self.logsumexp_PA(alpha_t_pa, pre_pa_path, axis=1), [self.batch_size, self.num_classes+1, 1])
            alphas_pa.append(alpha_t_pa)
            previous_pa = alpha_t_pa
            
        alphas = tf.reshape(tf.concat(alphas, 0), [self.num_steps + 2, self.batch_size, self.num_classes+1, 1])
        alphas = tf.transpose(alphas, [1, 0, 2, 3])
        alphas = tf.reshape(alphas, [self.batch_size * (self.num_steps + 2), self.num_classes+1, 1])

        last_alphas = tf.gather(alphas, tf.range(0, self.batch_size) * (self.num_steps + 2) + length)
        last_alphas = tf.reshape(last_alphas, [self.batch_size, self.num_classes+1, 1])
        
        ### pa add operation
        alphas_pa = tf.reshape(tf.concat(alphas_pa, 0), [self.num_steps + 2, self.batch_size, self.num_classes+1, 1])
        alphas_pa = tf.transpose(alphas_pa, [1, 0, 2, 3])
        alphas_pa = tf.reshape(alphas_pa, [self.batch_size * (self.num_steps + 2), self.num_classes+1, 1])
        
        last_alphas_pa = tf.gather(alphas_pa, tf.range(0, self.batch_size) * (self.num_steps + 2) + length)
        last_alphas_pa = tf.reshape(last_alphas_pa, [self.batch_size, self.num_classes+1, 1])

        max_scores = tf.reshape(tf.concat(max_scores, 0), (self.num_steps + 1, self.batch_size, self.num_classes+1))
        max_scores_pre = tf.reshape(tf.concat(max_scores_pre, 0), (self.num_steps + 1, self.batch_size, self.num_classes+1))
        max_scores = tf.transpose(max_scores, [1, 0, 2])
        max_scores_pre = tf.transpose(max_scores_pre, [1, 0, 2])

        return tf.reduce_sum(self.logsumexp_PA(last_alphas_pa, last_y_batch, axis=1)), tf.reduce_sum(self.logsumexp(last_alphas, axis=1)), max_scores, max_scores_pre
    
    
    '''
    * loss op supports to calculate crf loss by full annotation learning or partial annotation learning
    *     1. targets:
    *            pa: batch * num_steps * num_classes+1
    *            fa: batch * num_steps
    *     2. targets_transition: just needed for fa
    *
    '''
    def neg_log_likelihood_loss(self, targets, targets_transition, length):
        self._calculate_target_path_score(targets, targets_transition)
        self.total_path_score, self.max_scores, self.max_scores_pre = self.forward(self.observations, self.transitions, length)
        return - (self.target_path_score - self.total_path_score)
      
    
    def neg_log_likelihood_pa_loss(self, targets, length):
        self.PA_path_score, self.total_path_score, self.max_scores, self.max_scores_pre = self.PA_forward(self.observations, self.transitions, length, targets)
        # self.total_path_score, self.max_scores, self.max_scores_pre = self.forward(self.observations, self.transitions, length)
        return - (self.PA_path_score - self.total_path_score)
        
        
            
            
            
        