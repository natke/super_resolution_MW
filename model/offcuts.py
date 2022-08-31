        # Convert to YCbCr
        r: torch.Tensor = img[0, :, :]
        g: torch.Tensor = img[1, :, :]
        b: torch.Tensor = img[2, :, :]

        print(f'Red: {r}')
        print(f'Red: {g}')
        print(f'Red: {b}')

        delta = .5
        y: torch.Tensor = .299 * r + .587 * g + .114 * b
        cb: torch.Tensor = 128 - r * 0.169 - g * 0.331 + b * 0.5
        cr: torch.Tensor = 128 + r * 0.5 - g * 0.419 - b * 0.081 

        print(f'Y: {y}')
        print(f'Cb: {cb}')
        print(f'Cr: {cr}')
                              
        return y.unsqueeze_(0), cb.unsqueeze_(0), cr.unsqueeze_(0)
