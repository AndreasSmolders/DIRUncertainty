import torch.nn as nn
import torch
from torch.distributions import Normal
import numpy as np
from DL.Model.gaussianBlur import GaussianBlur
from Utils.constants import EPS
from Utils.interpolate import interpolateArray
from Utils.miscellaneous import roundUpToOdd

PROBABLITY_MODEL_DIAGONAL = "diagonal"
PROBABLITY_MODEL_BLOCK_DIAGONAL = "blockDiagonal"


class NormalSamplingLayer(nn.Module):
    def __init__(self, blurringLayer=None, nbSamples=1, downSampleFactor=1):
        super().__init__()
        self.fullResolutionBlurringLayer = blurringLayer
        self._setSampleDownSamplingFactor(downSampleFactor)
        self.nbSamples = nbSamples

    def forward(self, mean, stdev):
        if not self.training:
            return mean
        # print("IM DOING STUFF", mean)
        return mean + stdev * self.getStandardSample(mean.shape, mean.device)

    def getStandardSample(self, shape, device):
        dist = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        targetShape = self.getSampleShape(shape)
        eps1 = dist.rsample(targetShape)[..., 0].to(device)
        if self.blurringLayer is not None:
            eps1 = self.blurringLayer(eps1)
        eps1 = interpolateArray(eps1, shape[-3:])
        return self.downSampleConversionFactor * eps1

    def getSampleShape(self, shape):
        sampleShape = torch.tensor(shape)
        sampleShape[-3:] = torch.div(
            sampleShape[-3:], self.sampleDownSamplingFactor, rounding_mode="floor"
        )
        sampleShape[0] = sampleShape[0] * self.nbSamples
        if self.blurringLayer is not None and self.blurringLayer.padding == 0:
            sampleShape[-3:] = (
                sampleShape[-3:] + torch.tensor(self.blurringLayer.kernel_size) - 1
            )
        return sampleShape

    def _setSampleDownSamplingFactor(self, newDownSamplingFactor):
        self.sampleDownSamplingFactor = newDownSamplingFactor
        if self.fullResolutionBlurringLayer is not None:
            # Should maybe be in the blurring layer itself
            kernelSize = tuple(
                roundUpToOdd(
                    np.array(self.fullResolutionBlurringLayer.kernel_size)
                    // newDownSamplingFactor
                ).astype(np.int16)
            )
            smoothingSigma = tuple(
                np.array(self.fullResolutionBlurringLayer.sigma) / newDownSamplingFactor
            )
            self.blurringLayer = GaussianBlur(
                kernelSize,
                smoothingSigma,
                padding=self.fullResolutionBlurringLayer.padding,
                rescale=self.fullResolutionBlurringLayer.rescale,
            )
            self.downSampleConversionFactor = torch.sqrt(
                self.fullResolutionBlurringLayer.getVarianceWeights()
                / self.blurringLayer.getVarianceWeights()
            )
        else:
            self.blurringLayer = None
            self.downSampleConversionFactor = 1


