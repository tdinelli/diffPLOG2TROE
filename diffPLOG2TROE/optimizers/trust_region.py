def _trust_region_optimize(
    self,
    loss_fn: Callable,
    base_params: Array,
    active_indices: Array,
) -> Dict[str, Any]:
    """Optimize using a trust region algorithm."""
    # Reset state for this optimization run
    self.step = 0
    self.best_loss = jnp.float64("inf")
    self.best_params = base_params.copy()
    self.steps_without_improvement = 0

    # Extract active parameters to optimize
    active_params = base_params[active_indices]
    n = len(active_params)

    # Wrap the loss function to handle active parameters
    def loss_wrapper(active_params):
        full_params = base_params.at[active_indices].set(active_params)
        return loss_fn(full_params)

    # Initialize trust region radius
    trust_radius = self.initial_trust_radius

    # Compute initial loss and gradient
    loss_value, grad_value = value_and_grad(loss_wrapper)(active_params)

    # For trust region, we need Hessian or an approximation
    # Here we use a finite difference approximation for the Hessian
    def approx_hessian(params, grad_func):
        """Compute an approximate Hessian using finite differences on the gradient"""
        eps = 1e-6
        n = len(params)
        hess = jnp.zeros((n, n))

        # Compute gradient at the current point
        g0 = grad_func(params)

        # Compute approximate Hessian using finite differences
        for i in range(n):
            eps_vec = jnp.zeros(n)
            eps_vec = eps_vec.at[i].set(eps)
            g1 = grad_func(params + eps_vec)
            hess = hess.at[:, i].set((g1 - g0) / eps)

        # Ensure Hessian is symmetric
        hess = (hess + hess.T) / 2.0
        return hess

    # Gradient function
    grad_func = grad(loss_wrapper)

    # Record initial state
    current_params = active_params
    current_loss = loss_value
    current_grad = grad_value

    if current_loss < self.best_loss:
        self.best_loss = current_loss
        self.best_params = base_params.at[active_indices].set(current_params).copy()

    # Log initial state
    self._log(f"  Step 0: loss = {current_loss:.6e}, grad_norm = {jnp.linalg.norm(current_grad):.6e}")

    # Trust region main loop
    for step in range(1, self.max_steps + 1):
        self.step = step

        # Compute Hessian approximation
        hessian = approx_hessian(current_params, grad_func)

        # Solve the trust region subproblem
        # min_p  g^T p + 0.5 p^T H p  s.t. ||p|| <= trust_radius

        # Make Hessian positive definite if needed
        min_eig = jnp.min(jnp.linalg.eigvalsh(hessian))
        if min_eig <= 0:
            hessian = hessian + (abs(min_eig) + 1e-6) * jnp.eye(n)

        try:
            # Compute Newton step
            newton_step = -jnp.linalg.solve(hessian, current_grad)
            newton_step_norm = jnp.linalg.norm(newton_step)

            # If Newton step is within trust region, use it
            if newton_step_norm <= trust_radius:
                step_direction = newton_step
            else:
                # Otherwise, scale the step to the trust region boundary
                step_direction = (trust_radius / newton_step_norm) * newton_step
        except:
            # Fall back to steepest descent if numerical issues
            step_direction = -current_grad
            step_norm = jnp.linalg.norm(step_direction)
            if step_norm > 0:
                step_direction = (trust_radius / step_norm) * step_direction
            else:
                # Zero gradient - we're done
                break

        # Compute the predicted reduction from quadratic model
        pred_reduction = -jnp.dot(current_grad, step_direction) - 0.5 * jnp.dot(
            step_direction, jnp.matmul(hessian, step_direction)
        )

        # Compute new parameters and new loss
        new_params = current_params + step_direction
        new_loss = loss_wrapper(new_params)

        # Compute actual reduction
        actual_reduction = current_loss - new_loss

        # Compute ratio of actual to predicted reduction
        if abs(pred_reduction) < 1e-10:
            ratio = 1.0 if abs(actual_reduction) < 1e-10 else 0.0
        else:
            ratio = actual_reduction / pred_reduction

        # Update the trust region radius based on the ratio
        if ratio < self.eta1:
            # Poor agreement - shrink trust region
            trust_radius = max(self.gamma1 * trust_radius, 1e-10)
        elif ratio > self.eta2 and jnp.linalg.norm(step_direction) >= 0.9 * trust_radius:
            # Good agreement and step at boundary - expand trust region
            trust_radius = min(self.gamma2 * trust_radius, self.max_trust_radius)

        # Update parameters if we have improvement
        if actual_reduction > 0:
            current_params = new_params
            current_loss = new_loss
            current_grad = grad_func(current_params)

            # Track best parameters
            if current_loss < self.best_loss - self.early_stop_delta:
                self.best_loss = current_loss
                self.best_params = base_params.at[active_indices].set(current_params).copy()
                self.steps_without_improvement = 0
            else:
                self.steps_without_improvement += 1
        else:
            self.steps_without_improvement += 1

        # Log progress
        if step % self.log_interval == 0:
            self._log(
                f"  Step {step}: loss = {current_loss:.6e}, "
                f"grad_norm = {jnp.linalg.norm(current_grad):.6e}, "
                f"trust_radius = {trust_radius:.6e}, "
                f"ratio = {ratio:.6e}"
            )

        # Check early stopping conditions
        if self.steps_without_improvement >= self.early_stop_patience:
            self._log(f"Early stopping triggered after {step} steps")
            break

        if jnp.linalg.norm(current_grad) < self.early_stop_delta:
            self._log(f"Gradient norm below tolerance at step {step}")
            break

    # Return optimization results
    return {
        "params": self.best_params,
        "active_params": self.best_params[active_indices],
        "loss": self.best_loss,
        "iterations": self.step,
        "early_stopped": self.steps_without_improvement >= self.early_stop_patience,
    }
