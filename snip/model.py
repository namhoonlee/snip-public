import tensorflow as tf
import functools

from network import load_network


class Model(object):
    def __init__(self,
                 datasource,
                 arch,
                 num_classes,
                 target_sparsity,
                 optimizer,
                 lr_decay_type,
                 lr,
                 decay_boundaries,
                 decay_values,
                 initializer_w_bp,
                 initializer_b_bp,
                 initializer_w_ap,
                 initializer_b_ap,
                 **kwargs):
        self.datasource = datasource
        self.arch = arch
        self.num_classes = num_classes
        self.target_sparsity = target_sparsity
        self.optimizer = optimizer
        self.lr_decay_type = lr_decay_type
        self.lr = lr
        self.decay_boundaries = decay_boundaries
        self.decay_values = decay_values
        self.initializer_w_bp = initializer_w_bp
        self.initializer_b_bp = initializer_b_bp
        self.initializer_w_ap = initializer_w_ap
        self.initializer_b_ap = initializer_b_ap

    def construct_model(self):
        # Base-learner
        self.net = net = load_network(
            self.datasource, self.arch, self.num_classes,
            self.initializer_w_bp, self.initializer_b_bp,
            self.initializer_w_ap, self.initializer_b_ap,
        )

        # Input nodes
        self.inputs = net.inputs
        self.compress = tf.placeholder_with_default(False, [])
        self.is_train = tf.placeholder_with_default(False, [])
        self.pruned = tf.placeholder_with_default(False, [])

        # Switch for weights to use (before or after pruning)
        weights = tf.cond(self.pruned, lambda: net.weights_ap, lambda: net.weights_bp)

        # For convenience
        prn_keys = [k for p in ['w', 'b'] for k in weights.keys() if p in k]
        var_no_train = functools.partial(tf.Variable, trainable=False, dtype=tf.float32)

        # Model
        mask_init = {k: var_no_train(tf.ones(weights[k].shape)) for k in prn_keys}
        mask_prev = {k: var_no_train(tf.ones(weights[k].shape)) for k in prn_keys}

        def get_sparse_mask():
            w_mask = apply_mask(weights, mask_init)
            logits = net.forward_pass(w_mask, self.inputs['input'],
                self.is_train, trainable=False)
            loss = tf.reduce_mean(compute_loss(self.inputs['label'], logits))
            grads = tf.gradients(loss, [mask_init[k] for k in prn_keys])
            gradients = dict(zip(prn_keys, grads))
            cs = normalize_dict({k: tf.abs(v) for k, v in gradients.items()})
            return create_sparse_mask(cs, self.target_sparsity)

        mask = tf.cond(self.compress, lambda: get_sparse_mask(), lambda: mask_prev)
        with tf.control_dependencies([tf.assign(mask_prev[k], v) for k,v in mask.items()]):
            w_final = apply_mask(weights, mask)

        # Forward pass
        logits = net.forward_pass(w_final, self.inputs['input'], self.is_train)

        # Loss
        opt_loss = tf.reduce_mean(compute_loss(self.inputs['label'], logits))
        reg = 0.00025 * tf.reduce_sum([tf.reduce_sum(tf.square(v)) for v in w_final.values()])
        opt_loss = opt_loss + reg

        # Optimization
        optim, lr, global_step = prepare_optimization(opt_loss, self.optimizer, self.lr_decay_type,
            self.lr, self.decay_boundaries, self.decay_values)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # TF version issue
        with tf.control_dependencies(update_ops):
            self.train_op = optim.minimize(opt_loss, global_step=global_step)

        # Outputs
        output_class = tf.argmax(logits, axis=1, output_type=tf.int32)
        output_correct_prediction = tf.equal(self.inputs['label'], output_class)
        output_accuracy_individual = tf.cast(output_correct_prediction, tf.float32)
        output_accuracy = tf.reduce_mean(output_accuracy_individual)
        self.outputs = {
            'logits': logits,
            'los': opt_loss,
            'acc': output_accuracy,
            'acc_individual': output_accuracy_individual,
        }
        self.sparsity = compute_sparsity(w_final, prn_keys)

        # Summaries
        tf.summary.scalar('loss', opt_loss)
        tf.summary.scalar('accuracy', output_accuracy)
        tf.summary.scalar('lr', lr)
        self.summ_op = tf.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

def compute_loss(labels, logits):
    assert len(labels.shape)+1 == len(logits.shape)
    num_classes = logits.shape.as_list()[-1]
    labels = tf.one_hot(labels, num_classes, dtype=tf.float32)
    return tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)

def get_optimizer(optimizer, lr):
    if optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(lr)
    elif optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(lr, 0.9)
    else:
        raise NotImplementedError
    return optimizer

def prepare_optimization(loss, optimizer, lr_decay_type, learning_rate, boundaries, values):
    global_step = tf.Variable(0, trainable=False)
    if lr_decay_type == 'constant':
        learning_rate = tf.constant(learning_rate)
    elif lr_decay_type == 'piecewise':
        assert len(boundaries)+1 == len(values)
        learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)
    else:
        raise NotImplementedError
    optim = get_optimizer(optimizer, learning_rate)
    return optim, learning_rate, global_step

def vectorize_dict(x, sortkeys=None):
    assert isinstance(x, dict)
    if sortkeys is None:
        sortkeys = x.keys()
    def restore(v, x_shape, sortkeys):
        # v splits for each key
        split_sizes = []
        for key in sortkeys:
            split_sizes.append(functools.reduce(lambda x, y: x*y, x_shape[key]))
        v_splits = tf.split(v, num_or_size_splits=split_sizes)
        # x restore
        x_restore = {}
        for i, key in enumerate(sortkeys):
            x_restore.update({key: tf.reshape(v_splits[i], x_shape[key])})
        return x_restore
    # vectorized dictionary
    x_vec = tf.concat([tf.reshape(x[k], [-1]) for k in sortkeys], axis=0)
    # restore function
    x_shape = {k: x[k].shape.as_list() for k in sortkeys}
    restore_fn = functools.partial(restore, x_shape=x_shape, sortkeys=sortkeys)
    return x_vec, restore_fn

def normalize_dict(x):
    x_v, restore_fn = vectorize_dict(x)
    x_v_norm = tf.divide(x_v, tf.reduce_sum(x_v))
    x_norm = restore_fn(x_v_norm)
    return x_norm

def compute_sparsity(weights, target_keys):
    assert isinstance(weights, dict)
    w = {k: weights[k] for k in target_keys}
    w_v, _ = vectorize_dict(w)
    sparsity = tf.nn.zero_fraction(w_v)
    return sparsity

def create_sparse_mask(mask, target_sparsity):
    def threshold_vec(vec, target_sparsity):
        num_params = vec.shape.as_list()[0]
        kappa = int(round(num_params * (1. - target_sparsity)))
        topk, ind = tf.nn.top_k(vec, k=kappa, sorted=True)
        mask_sparse_v = tf.sparse_to_dense(ind, tf.shape(vec),
            tf.ones_like(ind, dtype=tf.float32), validate_indices=False)
        return mask_sparse_v
    if isinstance(mask, dict):
        mask_v, restore_fn = vectorize_dict(mask)
        mask_sparse_v = threshold_vec(mask_v, target_sparsity)
        return restore_fn(mask_sparse_v)
    else:
        return threshold_vec(mask, target_sparsity)

def apply_mask(weights, mask):
    all_keys = weights.keys()
    target_keys = mask.keys()
    remain_keys = list(set(all_keys) - set(target_keys))
    w_sparse = {k: mask[k] * weights[k] for k in target_keys}
    w_sparse.update({k: weights[k] for k in remain_keys})
    return w_sparse
