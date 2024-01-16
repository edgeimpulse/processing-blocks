# Edge Impulse processing blocks

These are officially supported processing blocks in Edge Impulse. These blocks can be selected from any project, and can be used as inspiration for [Building custom processing blocks](https://docs.edgeimpulse.com/docs/custom-blocks). These blocks are meant to be ran on a server. The corresponding C++ blocks are in the [C++ inferencing SDK](https://github.com/edgeimpulse/inferencing-sdk-cpp), or can be found in the edge-impulse-sdk folder of the zip exported from Studio.

## Contributing to this repository

We welcome contributions to this repository. Both improvements to our own processing blocks, as well as new and well-tested processing blocks for other sensor data. To contribute just open a pull request against this repository. Note that blocks require a corresponding implementation in the inferencing SDK. If you add options, change behavior, or add a new block please either open a corresponding pull request in the inferencing SDK, or ask for some help from the Edge Impulse team.

### Testing your contributions

The blocks in this repository are compatible with custom processing blocks. Follow the [Building custom processing blocks](https://docs.edgeimpulse.com/docs/custom-blocks) tutorial to learn how you can load modified processing blocks into Edge Impulse.
