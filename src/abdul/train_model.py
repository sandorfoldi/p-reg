import torch

# Train func
def train(loss_fn, model, data, mu, p_reg_dict=None, num_epochs=200):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    for i in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Clear gradients.
        Z = model(data)     # Perform a single forward pass.    
        loss = loss_fn(data, Z)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
    return loss, Z