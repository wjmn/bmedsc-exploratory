function result = gauss2d(x, y, x0, y0, sx, sy)
%GAUSS2D Returns the value at x of a Gaussian distribution with center x0y0
%and spread in the x and y direction of sx and sy respectively.
result = exp(-(((x-x0)^2 / (2 * sx^2)) + ((y - y0)^2 / (2 * sy^2))));
end

