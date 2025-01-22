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

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include <string.h>
#include "mtcnn.h"
#include "cnn.h"
#include "lite_face.h"

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

#define IMAGE_WIDTH 160
#define IMAGE_HEIGHT 120
#define CHANNEL_SIZE IMAGE_WIDTH*IMAGE_HEIGHT
#define MAX_DETECTED_FACES 10
#define MODE 1
#define TESTING 1

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;

TIM_HandleTypeDef htim1;

UART_HandleTypeDef huart2;
UART_HandleTypeDef huart3;

PCD_HandleTypeDef hpcd_USB_OTG_FS;

/* USER CODE BEGIN PV */

uint8_t rxBuffer[2] = {0};
volatile uint16_t rxBufferSize = 2;
volatile uint8_t rxInProgress = 0;
volatile uint8_t rxDone = 0;
volatile uint8_t firstPacket = 0;

uint8_t receivedData[CHANNEL_SIZE*2] = {0};
volatile size_t receivedDataSize = 0;
volatile uint16_t adcValue = 0;
volatile uint8_t abortReceivingFlag = 1;
volatile uint8_t timerTick = 0;
volatile uint8_t lastTimerTick = 0;

uint8_t txBufferGetImage[1] = {255};
uint8_t txBufferSentImage[1] = {254};
uint8_t txNull[5] = {0};

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART3_UART_Init(void);
static void MX_USB_OTG_FS_PCD_Init(void);
static void MX_USART2_UART_Init(void);
static void MX_ADC1_Init(void);
static void MX_TIM1_Init(void);
/* USER CODE BEGIN PFP */
void UART_SendArray(UART_HandleTypeDef *huart, const uint8_t *array, uint32_t arraySize, uint16_t chunkSize);
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

  /* Enable I-Cache---------------------------------------------------------*/
  SCB_EnableICache();

  /* Enable D-Cache---------------------------------------------------------*/
  SCB_EnableDCache();

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
  MX_ADC1_Init();
  MX_TIM1_Init();
  /* USER CODE BEGIN 2 */

  HAL_ADCEx_Calibration_Start(&hadc1, ADC_CALIB_OFFSET, ADC_SINGLE_ENDED);
  HAL_TIM_Base_Start_IT(&htim1);

  HAL_GPIO_WritePin(LD1_GPIO_Port, LD1_Pin, GPIO_PIN_SET);
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
	  if (rxDone){
		  // sending system mode
		  uint8_t modeTx[] = {MODE};
		  HAL_UART_Transmit(&huart2, modeTx, 1, 100);
		  HAL_GPIO_WritePin(PLED_GPIO_Port, PLED_Pin, GPIO_PIN_RESET);
		  HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, GPIO_PIN_RESET);
		  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);

		  // converting RGB5656 to RGB888
		  uint8_t imageRGB[CHANNEL_SIZE*3] = {0};
		  for (size_t i = 0, j = 0; i < receivedDataSize; i += 2, j += 1) {
			// read the RGB565 value (2 bytes)
			uint16_t pixel = (receivedData[i] << 8) | receivedData[i + 1];

			// extract RGB components from RGB565
			uint8_t r5 = (pixel >> 11) & 0x1F; // 5 bits for red
			uint8_t g6 = (pixel >> 5) & 0x3F;  // 6 bits for green
			uint8_t b5 = pixel & 0x1F;         // 5 bits for blue

			// convert to 8-bit per channel (RGB888)
			imageRGB[j] = (r5 << 3) | (r5 >> 2);	// red
			imageRGB[CHANNEL_SIZE + j] = (g6 << 2) | (g6 >> 4);	// green
			imageRGB[CHANNEL_SIZE*2 + j] = (b5 << 3) | (b5 >> 2);	// blue
		  }

		  // scaling image to 48x64
		  uint8_t scaledImage[3*48*64] = {0};
		  CNN_AdaptiveAveragePool_Uint8_Uint8(3, IMAGE_HEIGHT, IMAGE_WIDTH, 48, 64, imageRGB, scaledImage);

		  // face detection algorithm
		  float boxes[5*MAX_DETECTED_FACES];
		  int boxesLen = MTCNN_DetectFace(3, 48, 64, scaledImage, boxes);

		  if (TESTING){
			  uint8_t outputBuffer[boxesLen*5];
			  for (size_t i=0;i<boxesLen*5;++i){
				  outputBuffer[i] = roundf(boxes[i]);
			  }
			  HAL_UART_Transmit(&huart3, outputBuffer, boxesLen*5, 100);
			  if (boxesLen == 0){
				  HAL_UART_Transmit(&huart3, txNull, 5, 100);
			  }
		  }

		  // sending count of detected faces
		  uint8_t boxesLenTx[] = {(uint8_t)boxesLen};
		  HAL_UART_Transmit(&huart2, boxesLenTx, 1, 100);
		  if (boxesLen > 0){
			// scaling boxes to original size
			uint8_t finalBoxes[boxesLen*4];
			for (size_t i=0, j=0;i<boxesLen*5;i+=5, j+=4){
			  finalBoxes[j] = roundf(boxes[i] * 2.5f);
			  finalBoxes[j+1] = roundf(boxes[i+1] * 2.5f);
			  finalBoxes[j+2] = roundf(boxes[i+2] * 2.5f);
			  finalBoxes[j+3] = roundf(boxes[i+3] * 2.5f);
			}
			// sending detected boxes
			if (MODE == 0){
			  HAL_UART_Transmit(&huart2, finalBoxes, (uint16_t)boxesLen*4, 100);
			}
			else if (MODE == 1){
				// face recognition algorithm
				for (size_t i=0;i<boxesLen;++i){
					// extracting face from image
					size_t start = finalBoxes[i * 4] + finalBoxes[i * 4 + 1] * IMAGE_WIDTH;
					size_t width = finalBoxes[i * 4 + 2] - finalBoxes[i * 4];
					size_t height = finalBoxes[i * 4 + 3] - finalBoxes[i * 4 + 1];
					uint8_t alignedImage[3*height*width];
					for (uint8_t o=0;o<3;++o){
						for (size_t j=0; j<height;++j){
							memcpy(alignedImage + o*width*height + j*width, imageRGB + CHANNEL_SIZE*o + start + (j*IMAGE_WIDTH), width);
						}
					}
					if (TESTING){
						uint8_t txSize[] = {height, width};
						HAL_UART_Transmit(&huart3, txNull, 5, 100);
						HAL_UART_Transmit(&huart3, txSize, 2, 100);
//						HAL_UART_Transmit(&huart3, txNull, 5, 100);
//						UART_SendArray(&huart3, alignedImage, 3*height*width, 1024);
					}

					// scaling image to 100x100
					float scaledAlignedImage[3*100*100];
					CNN_AdaptiveAveragePool_Uint8_Float(3, height, width, 100, 100, alignedImage, scaledAlignedImage);
					if (TESTING){
						HAL_UART_Transmit(&huart3, txNull, 5, 100);
					}

					// image normalization
					for (size_t j=0;j<3*100*100;++j){
						scaledAlignedImage[j] /= 255;
					}
					const float means[3] = {0.5f, 0.5f, 0.5f};
					const float stds[3] = {0.5f, 0.5f, 0.5f};
					CNN_Normalize(3, 100, 100, scaledAlignedImage, means, stds);

					// getting face embedding
					float faceEmbedding[128];
					LiteFace_Model(scaledAlignedImage, faceEmbedding);

					// converting face embedding to raw bytes
					uint8_t txEmbedding[128*4] = {0};
					for (uint8_t j=0;j<128;++j){
						// copy float to uint8_t
						uint32_t temp = 0;
						memcpy(&temp, &faceEmbedding[j], sizeof(float));

						// convert uint32_t to 4 uint8_t (big-endian format)
						txEmbedding[j * 4] = (temp >> 24) & 0xFF;
						txEmbedding[j * 4 + 1] = (temp >> 16) & 0xFF;
						txEmbedding[j * 4 + 2] = (temp >> 8) & 0xFF;
						txEmbedding[j * 4 + 3] = temp & 0xFF;
					}

					// sending face embedding
					uint16_t chunkSize = 256;
					for (size_t pointer=0;pointer<4*128;pointer+=chunkSize){
						HAL_UART_Transmit(&huart2, txEmbedding+pointer, chunkSize, 1000);
						HAL_Delay(100);
					}
					if (TESTING){
						HAL_UART_Transmit(&huart3, txNull, 5, 100);
//						HAL_UART_Transmit(&huart3, txEmbedding, 512, 500);
					}
				}
			}
		  }
		  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
		  HAL_Delay(1000);
		  receivedDataSize = 0;
		  rxDone = 0;
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
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 23999;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 4999;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterOutputTrigger2 = TIM_TRGO2_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */

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
  lastTimerTick = timerTick;
  if (rxBufferSize == 2){
	  HAL_GPIO_WritePin(PLED_GPIO_Port, PLED_Pin, GPIO_PIN_RESET);
	  if (firstPacket){
		  firstPacket = 0;
		  HAL_UART_Transmit(&huart2, rxBuffer, 2, 100);
		  HAL_UART_Receive_IT(&huart2, rxBuffer, 2);
		  return;
	  }
	  rxBufferSize = ((uint16_t)rxBuffer[0] << 8) | (uint16_t)rxBuffer[1];
	  if (!rxBufferSize){
		  rxDone = 1;
		  rxInProgress = 0;
		  rxBufferSize = 2;
		  return;
	  }
	  HAL_UART_Receive_IT(&huart2, receivedData + receivedDataSize, rxBufferSize);
	  if (TESTING){
		  HAL_UART_Transmit(&huart3, rxBuffer, 2, 100);
	  }
  }
  else{
	  HAL_UART_Receive_IT(&huart2, rxBuffer, 2);
	  receivedDataSize += rxBufferSize;
	  rxBufferSize = 2;
  }
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
  if (rxInProgress || rxDone){
	  return;
  }
  lastTimerTick = timerTick;
  if(GPIO_Pin == GPIO_PIN_13) {
	  firstPacket = 1;
	  HAL_StatusTypeDef error = HAL_UART_Receive_IT(&huart2, rxBuffer, 2);
	  if (error != HAL_OK){
		  if (TESTING){
			  uint8_t txError[1] = {error};
			  HAL_UART_Transmit(&huart3, txError, 1, 100);
		  }
		  return;
	  }
	  HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, GPIO_PIN_SET);
	  HAL_UART_Transmit(&huart2, txBufferGetImage, 1, 100);
	  if (TESTING){
		  HAL_UART_Transmit(&huart3, txBufferGetImage, 1, 100);
	  }
	  rxInProgress = 1;
  }
  else if (GPIO_Pin == GPIO_PIN_8){
	  HAL_ADC_Start(&hadc1);
	  if (HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY) == HAL_OK) {
		  adcValue = HAL_ADC_GetValue(&hadc1);
		  if (adcValue <= 4096){
			  HAL_GPIO_WritePin(PLED_GPIO_Port, PLED_Pin, GPIO_PIN_SET);
		  }

		  if (TESTING){
			  uint8_t txADC[2] = {0};
			  txADC[0] = (adcValue >> 8) & 0xFF;
			  txADC[1] = adcValue & 0xFF;
			  HAL_UART_Transmit(&huart3, txADC, 2, 100);
		  }
	  }
	  firstPacket = 1;
	  HAL_StatusTypeDef error = HAL_UART_Receive_IT(&huart2, rxBuffer, 2);
	  if (error != HAL_OK){
		  if (TESTING){
			  uint8_t txError[1] = {error};
			  HAL_UART_Transmit(&huart3, txError, 1, 100);
		  }
		  return;
	  }
	  HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, GPIO_PIN_SET);
	  HAL_UART_Transmit(&huart2, txBufferGetImage, 1, 100);
	  if (TESTING){
		  HAL_UART_Transmit(&huart3, txBufferGetImage, 1, 100);
	  }
	  rxInProgress = 1;
  } else {
      __NOP();
  }
}

void HAL_UART_AbortReceiveCpltCallback(UART_HandleTypeDef *huart){
	rxDone = 0;
	rxInProgress = 0;
	abortReceivingFlag = 1;
    rxBufferSize = 2;
	HAL_GPIO_WritePin(PLED_GPIO_Port, PLED_Pin, GPIO_PIN_RESET);
	HAL_GPIO_WritePin(LD3_GPIO_Port, LD3_Pin, GPIO_PIN_RESET);
    if (TESTING){
    	uint8_t txAbortReceive[1] = {2};
		HAL_UART_Transmit(&huart3, txAbortReceive, 1, 100);
    }
}

void HAL_TIM_PeriodElapsedCallback(TIM_HandleTypeDef *htim)
{
    if (htim->Instance == TIM1)
    {
    	++timerTick;
    	if (TESTING){
            HAL_GPIO_TogglePin(LD1_GPIO_Port, LD1_Pin);
    	}
		if (timerTick - lastTimerTick >= 2 && rxInProgress && abortReceivingFlag){
		  abortReceivingFlag = 0;
		  HAL_UART_AbortReceive_IT(&huart2);
		}
    }
}

void UART_SendArray(UART_HandleTypeDef *huart, const uint8_t *array, uint32_t arraySize, uint16_t chunkSize){
	size_t pointer = 0;
	uint16_t txChunkSize = 0;
	for (size_t n = 0; n < arraySize; n += chunkSize){
	  if (n + chunkSize <= arraySize) {
		  txChunkSize = chunkSize;
	  } else {
		  txChunkSize = arraySize % chunkSize;
	  }
	  HAL_UART_Transmit(huart, array+pointer, txChunkSize, 1000);
	  pointer += txChunkSize;
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
