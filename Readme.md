# NLP Sentiment Analysis Pipeline December 2025

An end-to-end machine learning pipeline for financial sentiment analysis and leveraging news headlines to generate trading signals.

## Architecture Diagrams

### Current Implementation (Prototype Stage)
![Current NLP Pipeline](NLP%20Pipeline%20December%202025%20Final.png)
*Figure 1: Current prototype implementation showing active data collection and display components*

### Planned Full System
![Planned NLP Pipeline](NLP%20Pipeline%20Diagram%20December%202025.png)
*Figure 2: Complete planned architecture including trading components (not yet implemented)*

---

> ⚠️ **Project Status: Prototype Stage**  
> This project is actively being built and refined. The architecture and implementation details described here represent the planned system. **Currently in prototype stage**: We are only deploying the model with RSS feed integration for testing and evaluation. The trading components are NOT active and will not be implemented until the system undergoes significant fixes, changes, and validation. Once we establish a reliable data source and accumulate sufficient training data, we will implement proper live sentiment analysis and eventually activate trading capabilities.

## Overview

This system performs financial sentiment analysis using a pre-trained FinBERT transformer model. The pipeline collects news headlines from Reuters.com RSS feed for building a historical dataset. Live sentiment predictions are deployed on Investing.com via an RSS feed integration, which informs trading signal generation.

## Current Status

🚧 **Prototype Stage**: This pipeline is in early development focused on data collection and model evaluation.

**Active Components:**
- ✅ Using pre-trained FinBERT model (no custom training yet)
- ✅ Collecting headlines from Reuters.com for data accumulation
- ✅ Model deployed to Investing.com RSS feed for display and testing
- ✅ MongoDB storage infrastructure established

**Inactive Components (Not Yet Implemented):**
- ❌ Trading signal generation (framework planned but NOT active)
- ❌ Trading platform API integration (NOT implemented)
- ❌ Live trade execution (NOT active)
- ❌ Trade feedback loop (NOT implemented)

**In Progress:**
- 🔄 Building historical dataset for future model fine-tuning
- 🔄 Seeking reliable data platform/API for proper live sentiment analysis
- 🔄 Evaluating model performance through RSS feed deployment
- ⏳ Awaiting sufficient data to begin custom model training

**Trading Status**: The trading components shown in the architecture diagram are planned features but are **NOT currently active**. Trading functionality will only be implemented after extensive testing, validation, fixes, and system improvements.

## Architecture

The pipeline consists of six main components:

### 1. Scheduling and Automation
- **Monthly Airflow DAG Trigger**: Orchestrates data collection and storage
- Ensures consistent monthly data accumulation from Reuters.com RSS feed

### 2. Data Collection and Ingestion
- **Collect News Headlines**: Retrieves headlines from Reuters.com
- **Extract Headline, Date and Time**: Parses and structures RSS feed data
- **Store Data**: Saves processed headlines as CSV/Parquet files
- **Purpose**: Building historical dataset for future model training and analysis

### 3. Data Storage
- **Store Data as CSV/Parquet File**: Intermediate file storage for processing
- **Push Data to MongoDB**: Long-term storage in MongoDB on Docker/GCP
- **Building Dataset**: Creating historical archive for future model training

### 4. Model (Currently Using Pre-trained)
- **Live FinBERT Model with Standard Pretraining**: Using pre-trained FinBERT without custom fine-tuning
- **Future Training Pipeline**: Infrastructure ready for when sufficient labeled data is collected
- **Note**: Model training, validation, and testing will be implemented once we have adequate data

### 5. Model Deployment and Inference
- **Deploy Model to Investing.com RSS Feed**: Integration with Investing.com platform
- **Predict Headline Sentiment Live**: Real-time sentiment classification
- **Display Bullish or Bearish Status**: Visualizes market sentiment indicators on the platform

### 6. API and Signal Generation (Planned - Not Yet Active)
- **Send Signals to Trading Platform API**: *Planned integration with trading platforms*
- **Trade Execution**: *Will execute trades based on sentiment signals (future implementation)*
- **Feedback Loop**: *Will monitor trade performance (future implementation)*
- **⚠️ Note**: These components are part of the planned architecture but are **NOT currently implemented**. Trading functionality requires extensive testing, fixes, and validation before activation.

## Technology Stack

- **Orchestration**: Apache Airflow
- **Data Storage**: MongoDB (Docker/GCP), CSV/Parquet files
- **ML Model**: Pre-trained FinBERT (ProsusAI/finbert)
- **Data Sources**: 
  - Reuters.com RSS feed (historical data collection)
  - Investing.com RSS feed (live sentiment display and signal generation)
- **Data Processing**: Python, Pandas
- **Deployment**: Docker, GCP

## Data Sources & Their Uses

### Reuters.com RSS Feed
- **Purpose**: Historical data accumulation
- **Collection**: Monthly via Airflow
- **Storage**: MongoDB and CSV/Parquet files
- **Future Use**: Training dataset for model fine-tuning
- **Note**: Building dataset over time as we collect more headlines

### Investing.com RSS Feed
- **Purpose**: Live sentiment display and trading signals
- **Usage**: Platform for deploying model predictions
- **Output**: Real-time bullish/bearish predictions for trading
- **Status**: Currently active for signal generation




### Data Collection (Monthly)
1. **Airflow Trigger**: Monthly schedule initiates RSS feed collection from Reuters.com
2. **RSS Parsing**: Headlines extracted with timestamps
3. **File Storage**: Data saved as CSV/Parquet files
4. **Database Storage**: Headlines pushed to MongoDB
5. **Purpose**: Building historical dataset for future model training

### Live Sentiment Display (Prototype - Current)
1. **Pre-trained FinBERT**: Analyzes sentiment without custom training
2. **Investing.com Deployment**: Model predictions displayed on RSS feed
3. **Evaluation**: Monitoring prediction quality and gathering insights

### Trading Pipeline (Planned - Not Active)
*The following workflow is planned but NOT currently active:*
1. ~~Signal Generation: Trading signals created based on sentiment predictions~~
2. ~~API Integration: Signals sent to trading platform~~
3. ~~Execution & Feedback: Trades executed and performance monitored~~

**Note**: Trading components will only be activated after significant development, testing, and validation phases are complete.

## Model Details

The pipeline uses **pre-trained FinBERT** (ProsusAI/finbert), a BERT-based model pre-trained for financial sentiment analysis. The model classifies headlines into:
- **Bullish** (positive market sentiment)
- **Bearish** (negative market sentiment)
- **Neutral** (no clear direction)

**Current Approach**: Using the model as-is without additional fine-tuning

**Future Plans**: Once we have:
1. A reliable and solid data platform/API
2. Sufficient labeled headline data
3. Established data collection pipeline

We will implement proper live sentiment analysis with custom model training for improved accuracy.

## Data Flow

### Data Collection Flow
```
Reuters.com RSS → Airflow (Monthly) → Parse Headlines → 
CSV/Parquet Files → MongoDB → Historical Dataset
```

### Trading Signal Flow (Planned - Not Active)
```
MongoDB Data → Pre-trained FinBERT → Sentiment Prediction → 
Investing.com RSS Feed (Display Only)

[Future Implementation]:
→ Trading Signals → Platform API → Execution → Feedback
```

**Current State**: Only the display component is active. Trading signal generation, API integration, and execution are planned but not implemented.

## Monitoring and Maintenance

**Currently Monitored:**
- Airflow DAG execution for monthly data collection
- Data accumulation in MongoDB
- Sentiment predictions on Investing.com RSS feed
- Model prediction quality and consistency

**Future Monitoring (When Trading is Active):**
- Trading signal accuracy
- Trade execution performance
- Feedback loop metrics
- Risk management indicators

## Known Limitations & Challenges

**Prototype Stage Limitations:**
1. **Trading Not Active**: Trading components are planned but NOT implemented - this is display/testing only
2. **Extensive Work Required**: Trading functionality requires significant fixes, changes, and validation before activation
3. **Pre-trained Model Only**: Currently using FinBERT without custom training or domain-specific fine-tuning

**Technical Challenges:**
4. **Data Platform Search**: Actively seeking reliable data platform/API for proper live sentiment analysis
5. **RSS Feed Limitations**: Current RSS feeds may have delays or limited coverage
6. **Data Collection Phase**: Still accumulating data needed for custom model training
7. **Model Optimization**: Pre-trained model not yet optimized for specific trading strategies
8. **System Validation**: Requires extensive testing before any trading implementation

## Roadmap

### Phase 1: Prototype & Data Foundation (Current)
- [x] Implement pre-trained FinBERT model
- [x] Set up Reuters.com RSS data collection via Airflow
- [x] Deploy model to Investing.com RSS feed (display only)
- [x] Establish MongoDB storage infrastructure
- [ ] Evaluate model predictions through RSS feed deployment
- [ ] Identify and integrate reliable data platform/API
- [ ] Accumulate 6+ months of headline data
- [ ] Implement data labeling workflow
- [ ] Document required fixes and changes for trading implementation

### Phase 2: System Refinement & Model Training
- [ ] Label collected historical data
- [ ] Prepare training/validation/test datasets
- [ ] Fine-tune FinBERT on collected dataset
- [ ] Validate and test custom model
- [ ] A/B test pre-trained vs fine-tuned model performance
- [ ] Implement comprehensive logging and monitoring
- [ ] Address system issues identified in prototype phase

### Phase 3: Trading Implementation Preparation
- [ ] Design and document trading logic and risk management
- [ ] Implement paper trading environment for testing
- [ ] Develop signal generation algorithms
- [ ] Create backtesting framework with historical data
- [ ] Extensive testing of trading signals without real execution
- [ ] Implement safety mechanisms and kill switches
- [ ] Performance validation over extended test period

### Phase 4: Live Trading (Future)
- [ ] Implement proper live sentiment analysis with reliable data source
- [ ] Deploy custom-trained model to production
- [ ] Activate trading platform API integration
- [ ] Begin with minimal capital for live testing
- [ ] Implement advanced risk management features
- [ ] Real-time monitoring and alerting system
- [ ] Gradual scale-up based on performance

### Phase 5: Expansion (Long-term)
- [ ] Multi-source news aggregation
- [ ] Automated data labeling pipeline
- [ ] Sentiment trend analysis over time
- [ ] Ensemble models combining multiple sentiment sources
- [ ] Enhanced trading strategies based on historical performance

## Future Enhancements

- [ ] Establish connection to professional financial data API
- [ ] Real-time data ingestion pipeline
- [ ] Implement active learning for efficient data labeling
- [ ] Build confidence scoring system for trading signals
- [ ] Add market context awareness (sector news, economic indicators)
- [ ] Develop automated retraining pipeline
- [ ] Create comprehensive backtesting framework

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

⚠️ **Important**: 

**Prototype Status**: This system is currently in the prototype stage with **NO ACTIVE TRADING COMPONENTS**. Only the RSS feed display functionality is operational.

**Not for Trading**: Do NOT attempt to use this system for any form of trading, paper or live. The trading components shown in the architecture are planned features that require extensive development, testing, fixes, and validation before they can be safely implemented.

**Educational Purpose**: This project is for educational and research purposes only. Trading financial instruments carries significant risk of loss.

**Development Required**: The trading functionality requires substantial additional work including but not limited to:
- System stability improvements
- Comprehensive testing and validation
- Risk management implementation
- Performance optimization
- Regulatory compliance review
- Professional review and audit

**Future Implementation**: Even when trading components are developed, always consult with financial advisors before making any investment decisions.

**Use at Your Own Risk**: Any use of this code or system is entirely at your own risk.

## Project Notes

**Current Prototype Scope:**

This pipeline represents our planned architecture for a comprehensive financial sentiment analysis and trading system. However, we are currently in the **prototype stage** with a limited scope:

**What's Active Now:**
- Data collection from Reuters.com RSS feed
- Storage in MongoDB and file systems
- Pre-trained FinBERT model deployment
- Sentiment display on Investing.com RSS feed

**What's NOT Active:**
- Trading signal generation
- Trading platform API integration
- Trade execution
- Feedback loops
- Any form of automated trading

**Development Path:**

As we progress through the development phases, our priorities are:
1. **Data Foundation**: Secure reliable data sources and accumulate training data
2. **Model Training**: Develop and validate custom-trained models
3. **System Refinement**: Address bugs, improve stability, and implement proper monitoring
4. **Testing Phase**: Extensive testing without real trading (paper trading, backtesting)
5. **Trading Implementation**: Only after all above phases are complete and validated

**Why Trading Isn't Active:**

The trading components require significant additional work including:
- Comprehensive system testing and validation
- Bug fixes and stability improvements
- Risk management implementation
- Backtesting framework development
- Paper trading validation
- Performance optimization
- Safety mechanisms and fail-safes
- Legal and regulatory compliance review



# Need Help?
📺 Video Tutorials & Assistance
For additional help, tutorials, and guidance on this project, visit my YouTube channel:
https://www.youtube.com/@BDB5905

Find video walkthroughs, setup guides, troubleshooting tips, and project updates to help you work with this NLP pipeline.
