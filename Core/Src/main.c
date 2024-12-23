/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "string.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

#include "mtcnn.h"
#include "inputs.h"
#include "cnn.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#define ENABLE_TEST_PRINT 1
#define RX_MAX_BUFF_SIZE 1024
#define IMAGE_WIDTH 160
#define IMAGE_HEIGHT 120
#define CHANNEL_SIZE IMAGE_WIDTH*IMAGE_HEIGHT

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
#if defined ( __ICCARM__ ) /*!< IAR Compiler */
#pragma location=0x30040000
ETH_DMADescTypeDef  DMARxDscrTab[ETH_RX_DESC_CNT]; /* Ethernet Rx DMA Descriptors */
#pragma location=0x30040060
ETH_DMADescTypeDef  DMATxDscrTab[ETH_TX_DESC_CNT]; /* Ethernet Tx DMA Descriptors */

#elif defined ( __CC_ARM )  /* MDK ARM Compiler */

__attribute__((at(0x30040000))) ETH_DMADescTypeDef  DMARxDscrTab[ETH_RX_DESC_CNT]; /* Ethernet Rx DMA Descriptors */
__attribute__((at(0x30040060))) ETH_DMADescTypeDef  DMATxDscrTab[ETH_TX_DESC_CNT]; /* Ethernet Tx DMA Descriptors */

#elif defined ( __GNUC__ ) /* GNU Compiler */
ETH_DMADescTypeDef DMARxDscrTab[ETH_RX_DESC_CNT] __attribute__((section(".RxDecripSection"))); /* Ethernet Rx DMA Descriptors */
ETH_DMADescTypeDef DMATxDscrTab[ETH_TX_DESC_CNT] __attribute__((section(".TxDecripSection")));   /* Ethernet Tx DMA Descriptors */

#endif

ETH_TxPacketConfig TxConfig;

ADC_HandleTypeDef hadc1;

ETH_HandleTypeDef heth;

JPEG_HandleTypeDef hjpeg;

UART_HandleTypeDef huart2;
UART_HandleTypeDef huart3;

PCD_HandleTypeDef hpcd_USB_OTG_FS;

/* USER CODE BEGIN PV */

uint8_t rxBuffer[RX_MAX_BUFF_SIZE] = {0};
volatile uint8_t rxDone = 0;
volatile uint8_t rxInProgress = 0;
volatile uint16_t rxBufferSize = 2;
uint8_t txBufferGetImage[1] = {255};
uint8_t txBufferSentImage[1] = {0};
uint8_t txBufferTest[1] = {1};
uint8_t enableEcho = 0;
uint8_t receivedData[CHANNEL_SIZE*2] = {0};
//uint8_t receivedData[10000] = {0};
volatile size_t receivedDataSize = 0;
uint8_t txNull[5] = {0};

//uint8_t image[320*240*3] = {0};
//uint32_t imageSize = 320*240*3;

volatile uint16_t adcValue = 0;

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_USB_OTG_FS_PCD_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_JPEG_Init(void);
static void MX_ADC1_Init(void);
static void MX_ETH_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART3_UART_Init();
  MX_USB_OTG_FS_PCD_Init();
  MX_USART2_UART_Init();
  MX_JPEG_Init();
  MX_ADC1_Init();
  MX_ETH_Init();
  /* USER CODE BEGIN 2 */

  HAL_ADCEx_Calibration_Start(&hadc1, ADC_CALIB_OFFSET, ADC_SINGLE_ENDED);

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
//      // Wait for ADC conversion to complete
//	  HAL_ADC_Start(&hadc1);
//      if (HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY) == HAL_OK) {
//          // Get the ADC value
//    	  adcValue = HAL_ADC_GetValue(&hadc1);
//
//          // Use adcValue as needed
//          uint8_t txADC[2] = {0};
//    	  txADC[0] = (adcValue >> 8) & 0xFF;  // Extract the higher 8 bits
//    	  txADC[1] = adcValue & 0xFF;         // Extract the lowest 8 bits
//    	  HAL_UART_Transmit(&huart3, txADC, 2, 100);
//    	  if (adcValue <= 255){
//    		  HAL_GPIO_WritePin(PLED_GPIO_Port, PLED_Pin, GPIO_PIN_SET);
//    	  }
//    	  else{
//    		  HAL_GPIO_WritePin(PLED_GPIO_Port, PLED_Pin, GPIO_PIN_RESET);
//    	  }
//    	  HAL_Delay(2000);
//      }

	  if (rxDone){
		  HAL_GPIO_WritePin(PLED_GPIO_Port, PLED_Pin, GPIO_PIN_RESET);
//		  HAL_UART_Transmit(&huart3, txNull, 5, 100);
//		  HAL_UART_Transmit(&huart3, receivedData, receivedDataSize, 5000);
//		  size_t pointer = 0;
//		  HAL_UART_Transmit(&huart3, txNull, 5, 100);
//		  uint16_t txChunkSize = 0;
//		  for (size_t n = 0; n < receivedDataSize; n += 1024){
//			  if (n + 1024 <= receivedDataSize) {
//				  txChunkSize = 1024;
//			  } else {
//				  txChunkSize = receivedDataSize % 1024;
//			  }
//			  HAL_UART_Transmit(&huart3, receivedData+pointer, txChunkSize, 1000);
//			  pointer += txChunkSize; // Move pointer by size
//			}

		  uint8_t imageRGB[CHANNEL_SIZE*3] = {0};
		  uint32_t imageRGBSize = CHANNEL_SIZE*3;
		  for (size_t i = 0, j = 0; i < receivedDataSize; i += 2, j += 1) {
			// Read the RGB565 value (2 bytes)
			uint16_t pixel = (receivedData[i] << 8) | receivedData[i + 1];

			// Extract RGB components from RGB565
			uint8_t r5 = (pixel >> 11) & 0x1F; // 5 bits for red
			uint8_t g6 = (pixel >> 5) & 0x3F;  // 6 bits for green
			uint8_t b5 = pixel & 0x1F;         // 5 bits for blue

			// Convert to 8-bit per channel (RGB888)
			imageRGB[j] = (r5 << 3) | (r5 >> 2);       // Red
			imageRGB[CHANNEL_SIZE + j] = (g6 << 2) | (g6 >> 4);   // Green
			imageRGB[CHANNEL_SIZE*2 + j] = (b5 << 3) | (b5 >> 2);   // Blue
		  }
//			for (size_t i = 0, j = 0; i < receivedDataSize; i += 2, j += 3) {
//				// Read the RGB565 value (2 bytes)
//				uint16_t pixel = (receivedData[i] << 8) | receivedData[i + 1];
//
//				// Extract RGB components from RGB565
//				uint8_t r5 = (pixel >> 11) & 0x1F; // 5 bits for red
//				uint8_t g6 = (pixel >> 5) & 0x3F;  // 6 bits for green
//				uint8_t b5 = pixel & 0x1F;         // 5 bits for blue
//
//				// Convert to 8-bit per channel (RGB888)
//				imageRGB[j] = (r5 << 3) | (r5 >> 2);       // Red
//				imageRGB[j+1] = (g6 << 2) | (g6 >> 4);   // Green
//				imageRGB[j+2] = (b5 << 3) | (b5 >> 2);   // Blue
//			}

//		  size_t pointer = 0;
//		  uint16_t txChunkSize = 0;
//		  for (size_t n = 0; n < imageRGBSize; n += 1024){
//			  if (n + 1024 <= imageRGBSize) {
//				  txChunkSize = 1024;
//			  } else {
//				  txChunkSize = imageRGBSize % 1024;
//			  }
//			  HAL_UART_Transmit(&huart3, imageRGB+pointer, txChunkSize, 1000);
//			  pointer += txChunkSize; // Move pointer by size
//			}

		  uint8_t scaledImage[3*48*64] = {0};
		  CNN_AdaptiveAveragePool_Uint8_Uint8(3, IMAGE_HEIGHT, IMAGE_WIDTH, 48, 64, imageRGB, scaledImage);

		  float boxes[5*10];
//		  int boxesLen = MTCNN_DetectFace(3, 50, 50, detectFaceInput, boxes);
//		  int boxesLen = MTCNN_DetectFace(3, 120, 160, detectFaceInput120_160, boxes);
//		  int boxesLen = MTCNN_DetectFace(3, 75, 75, detectFaceInput100_100, boxes);
//		  int boxesLen = MTCNN_DetectFace(3, IMAGE_HEIGHT, IMAGE_WIDTH, imageRGB, boxes);
		  int boxesLen = MTCNN_DetectFace(3, 48, 64, scaledImage, boxes);
		  uint8_t outputBuffer[boxesLen*5];
		  for (size_t i=0;i<boxesLen*5;++i){
			  outputBuffer[i] = (int) roundf(boxes[i]);
		  }
		  HAL_UART_Transmit(&huart3, outputBuffer, boxesLen*5, 1000);
		  if (boxesLen == 0){
			  HAL_UART_Transmit(&huart3, txNull, 5, 1000);
		  }
		  uint8_t finalBoxes[boxesLen*4];
		  for (size_t i = 0, j =0;i<boxesLen*5;i+=5, j+=4){
			  finalBoxes[j] = roundf(boxes[i] * 2.5f);
			  finalBoxes[j+1] = roundf(boxes[i+1] * 2.5f);
			  finalBoxes[j+2] = roundf(boxes[i+2] * 2.5f);
			  finalBoxes[j+3] = roundf(boxes[i+3] * 2.5f);
		  }
		  uint8_t boxesLenTx[] = {(uint8_t)boxesLen};
		  HAL_UART_Transmit(&huart2, boxesLenTx, 1, 100);
		  HAL_UART_Transmit(&huart2, finalBoxes, (uint16_t)boxesLen*4, 100);

//		  HAL_StatusTypeDef status = HAL_JPEG_Decode(&hjpeg, receivedData, receivedDataSize, image, imageSize, HAL_MAX_DELAY);
//		  if (status != HAL_OK) {
//			  uint8_t txError[1] = {status};
//			  HAL_UART_Transmit(&huart3, txError, 1, 100);
//		  }
//		  else{
//			  size_t imagePointer = 0;
//			  HAL_UART_Transmit(&huart3, txNull, 5, 100);
//			  uint16_t txChunkSize = 0;
//			  for (size_t n = 0; n < imageSize; n += 1024){
//			      if (n + 1024 <= imageSize) {
//			    	  txChunkSize = 1024;
//			      } else {
//			    	  txChunkSize = imageSize % 1024;
//			      }
//				  HAL_UART_Transmit(&huart3, image+imagePointer, txChunkSize, 1000);
//				  imagePointer += txChunkSize; // Move pointer by size
//			    }
//		  }
		  rxDone = 0;
		  receivedDataSize = 0;
	  }
	  if (enableEcho){
		  HAL_StatusTypeDef error = HAL_UART_Receive(&huart2, rxBuffer, 1, 100);
		  if (error == HAL_OK ){
			  HAL_UART_Transmit(&huart3, rxBuffer, 1, 100);
		  }
	  }
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE2);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI48|RCC_OSCILLATORTYPE_HSI
                              |RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.HSI48State = RCC_HSI48_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 1;
  RCC_OscInitStruct.PLL.PLLN = 30;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 4;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_3;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV1;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_MultiModeTypeDef multimode = {0};
  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Common config
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ClockPrescaler = ADC_CLOCK_ASYNC_DIV256;
  hadc1.Init.Resolution = ADC_RESOLUTION_16B;
  hadc1.Init.ScanConvMode = ADC_SCAN_DISABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc1.Init.LowPowerAutoWait = DISABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ConversionDataManagement = ADC_CONVERSIONDATA_DR;
  hadc1.Init.Overrun = ADC_OVR_DATA_PRESERVED;
  hadc1.Init.LeftBitShift = ADC_LEFTBITSHIFT_NONE;
  hadc1.Init.OversamplingMode = DISABLE;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the ADC multi-mode
  */
  multimode.Mode = ADC_MODE_INDEPENDENT;
  if (HAL_ADCEx_MultiModeConfigChannel(&hadc1, &multimode) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_2;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_1CYCLE_5;
  sConfig.SingleDiff = ADC_SINGLE_ENDED;
  sConfig.OffsetNumber = ADC_OFFSET_NONE;
  sConfig.Offset = 0;
  sConfig.OffsetSignedSaturation = DISABLE;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief ETH Initialization Function
  * @param None
  * @retval None
  */
static void MX_ETH_Init(void)
{

  /* USER CODE BEGIN ETH_Init 0 */

  /* USER CODE END ETH_Init 0 */

   static uint8_t MACAddr[6];

  /* USER CODE BEGIN ETH_Init 1 */

  /* USER CODE END ETH_Init 1 */
  heth.Instance = ETH;
  MACAddr[0] = 0x00;
  MACAddr[1] = 0x80;
  MACAddr[2] = 0xE1;
  MACAddr[3] = 0x00;
  MACAddr[4] = 0x00;
  MACAddr[5] = 0x00;
  heth.Init.MACAddr = &MACAddr[0];
  heth.Init.MediaInterface = HAL_ETH_RMII_MODE;
  heth.Init.TxDesc = DMATxDscrTab;
  heth.Init.RxDesc = DMARxDscrTab;
  heth.Init.RxBuffLen = 1524;

  /* USER CODE BEGIN MACADDRESS */

  /* USER CODE END MACADDRESS */

  if (HAL_ETH_Init(&heth) != HAL_OK)
  {
    Error_Handler();
  }

  memset(&TxConfig, 0 , sizeof(ETH_TxPacketConfig));
  TxConfig.Attributes = ETH_TX_PACKETS_FEATURES_CSUM | ETH_TX_PACKETS_FEATURES_CRCPAD;
  TxConfig.ChecksumCtrl = ETH_CHECKSUM_IPHDR_PAYLOAD_INSERT_PHDR_CALC;
  TxConfig.CRCPadCtrl = ETH_CRC_PAD_INSERT;
  /* USER CODE BEGIN ETH_Init 2 */

  /* USER CODE END ETH_Init 2 */

}

/**
  * @brief JPEG Initialization Function
  * @param None
  * @retval None
  */
static void MX_JPEG_Init(void)
{

  /* USER CODE BEGIN JPEG_Init 0 */

  /* USER CODE END JPEG_Init 0 */

  /* USER CODE BEGIN JPEG_Init 1 */

  /* USER CODE END JPEG_Init 1 */
  hjpeg.Instance = JPEG;
  if (HAL_JPEG_Init(&hjpeg) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN JPEG_Init 2 */

  /* USER CODE END JPEG_Init 2 */

}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  huart2.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart2.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart2.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart2, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart2, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart3.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart3, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart3, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * @brief USB_OTG_FS Initialization Function
  * @param None
  * @retval None
  */
static void MX_USB_OTG_FS_PCD_Init(void)
{

  /* USER CODE BEGIN USB_OTG_FS_Init 0 */

  /* USER CODE END USB_OTG_FS_Init 0 */

  /* USER CODE BEGIN USB_OTG_FS_Init 1 */

  /* USER CODE END USB_OTG_FS_Init 1 */
  hpcd_USB_OTG_FS.Instance = USB_OTG_FS;
  hpcd_USB_OTG_FS.Init.dev_endpoints = 9;
  hpcd_USB_OTG_FS.Init.speed = PCD_SPEED_FULL;
  hpcd_USB_OTG_FS.Init.dma_enable = DISABLE;
  hpcd_USB_OTG_FS.Init.phy_itface = PCD_PHY_EMBEDDED;
  hpcd_USB_OTG_FS.Init.Sof_enable = ENABLE;
  hpcd_USB_OTG_FS.Init.low_power_enable = DISABLE;
  hpcd_USB_OTG_FS.Init.lpm_enable = DISABLE;
  hpcd_USB_OTG_FS.Init.battery_charging_enable = ENABLE;
  hpcd_USB_OTG_FS.Init.vbus_sensing_enable = ENABLE;
  hpcd_USB_OTG_FS.Init.use_dedicated_ep1 = DISABLE;
  if (HAL_PCD_Init(&hpcd_USB_OTG_FS) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USB_OTG_FS_Init 2 */

  /* USER CODE END USB_OTG_FS_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOF_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOG_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, LD1_Pin|LD3_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(USB_OTG_FS_PWR_EN_GPIO_Port, USB_OTG_FS_PWR_EN_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(PLED_GPIO_Port, PLED_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : LD1_Pin LD3_Pin */
  GPIO_InitStruct.Pin = LD1_Pin|LD3_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

  /*Configure GPIO pin : USB_OTG_FS_PWR_EN_Pin */
  GPIO_InitStruct.Pin = USB_OTG_FS_PWR_EN_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(USB_OTG_FS_PWR_EN_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : USB_OTG_FS_OVCR_Pin */
  GPIO_InitStruct.Pin = USB_OTG_FS_OVCR_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(USB_OTG_FS_OVCR_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : PIR_Pin */
  GPIO_InitStruct.Pin = PIR_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  HAL_GPIO_Init(PIR_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : PLED_Pin */
  GPIO_InitStruct.Pin = PLED_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_PULLDOWN;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(PLED_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI9_5_IRQn, 1, 0);
  HAL_NVIC_EnableIRQ(EXTI9_5_IRQn);

  HAL_NVIC_SetPriority(EXTI15_10_IRQn, 1, 0);
  HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if (rxBufferSize == 2){
	  rxBufferSize = ((uint16_t)rxBuffer[0] << 8) | (uint16_t)rxBuffer[1];
	  if (!rxBufferSize){
		  rxDone = 1;
		  rxInProgress = 0;
		  rxBufferSize = 2;
		  return;
	  }
	  HAL_UART_Receive_IT(&huart2, receivedData + receivedDataSize, rxBufferSize);
	  HAL_UART_Transmit(&huart3, rxBuffer, 2, 100);
  }
  else{
//	  memcpy(receivedData + receivedDataSize, rxBuffer, rxBufferSize);
	  HAL_UART_Receive_IT(&huart2, rxBuffer, 2);
	  receivedDataSize += rxBufferSize;
	  rxBufferSize = 2;
  }
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
  if(GPIO_Pin == GPIO_PIN_13) {
	  if (!rxInProgress){
		  HAL_StatusTypeDef error = HAL_UART_Receive_IT(&huart2, rxBuffer, rxBufferSize);
		  if (error == HAL_ERROR || error == HAL_BUSY){
			  uint8_t txError[1] = {error};
			  HAL_UART_Transmit(&huart3, txError, 1, 100);
			  return;
		  }
		  HAL_UART_Transmit(&huart2, txBufferGetImage, 1, 100);
		  HAL_UART_Transmit(&huart3, txBufferGetImage, 1, 100);
		  rxInProgress = 1;
	  }
  }
  else if (GPIO_Pin == GPIO_PIN_8){
	  if (!rxInProgress){
		  HAL_ADC_Start(&hadc1);
	      if (HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY) == HAL_OK) {
	          // Get the ADC value
	    	  adcValue = HAL_ADC_GetValue(&hadc1);
	    	  if (adcValue <= 255){
	    		  HAL_GPIO_WritePin(PLED_GPIO_Port, PLED_Pin, GPIO_PIN_SET);
	    	  }

	          // Use adcValue as needed
	          uint8_t txADC[2] = {0};
	    	  txADC[0] = (adcValue >> 8) & 0xFF;  // Extract the higher 8 bits
	    	  txADC[1] = adcValue & 0xFF;         // Extract the lowest 8 bits
	    	  HAL_UART_Transmit(&huart3, txADC, 2, 100);
	      }
		  HAL_StatusTypeDef error = HAL_UART_Receive_IT(&huart2, rxBuffer, rxBufferSize);
		  if (error == HAL_ERROR || error == HAL_BUSY){
			  uint8_t txError[1] = {error};
			  HAL_UART_Transmit(&huart3, txError, 1, 100);
			  return;
		  }
		  HAL_UART_Transmit(&huart2, txBufferSentImage, 1, 100);
		  HAL_UART_Transmit(&huart3, txBufferSentImage, 1, 100);
		  rxInProgress = 1;
	  }
  } else {
      __NOP();
  }
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
