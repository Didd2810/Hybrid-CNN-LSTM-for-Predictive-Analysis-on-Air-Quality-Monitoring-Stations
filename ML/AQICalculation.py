
class AQI_calc:
    def __init__(self):
        self.aqi_categories = [(0, 50, 'Good'), (51, 100, 'Moderate'), (101, 150, 'Unhealthy for Sensitive Groups'), 
                          (151, 200, 'Unhealthy'), (201, 300, 'Very Unhealthy'), (301, 500, 'Hazardous')]
        self.co_8_breakpoint = [(0,5499), (5500,10499), (10500,14499), (14500,17999), (18000, 35499), (35500, 58499)]
        self.o3_1_breakpoint = [(), (), (200,322.99), (323,400.99), (401,792.99), (793,1184.99)]
        self.o3_8_breakpoint = [(0,100.99), (101,200.99), (121,167.99), (168,206.99), (207, 392.99), ()]
        self.so2_1_breakpoint = [(0,92.99), (93,350.99), (351,485.99), (486,797.99), (), ()]
        self.so2_24_breakpoint = [(0,0.99), (-1,-1), (-1,-1), (-1,-1), (798,1583.99), (1584,2631.99)]
        self.no2_24_breakpoint = [(0,100.99), (101,400.99), (401,677.99), (678,1221.99), (1222, 2349.99), (2350,3853.99)]
        self.pm10_24_breakpoint = [(0,75.99), (76,150.99), (151,250.99), (251,350.99), (351,420.99), (421,2000)]
        self.pm2_5_24_breakpoint = [(0.0, 50.49), (50.5,60.49), (60.5,75.49), (75.5,150.49), (150.5,250.49), (250.5,2000)]
        
        
    def hour_avg(self, hours, gas_num, data_set):
        moving_avg = []
        
        for i in range(hours-1):
            moving_avg.append(data_set[i])
        for i in range((len(data_set)-hours+1)):
            sel_data = data_set[i:len(data_set)]
            sum=0
            avg = 0
            for q in range(hours):
                
                sum+=sel_data[q]
            avg = sum/hours
            if(gas_num=='NO2' and gas_num=='SO2' and gas_num=='PM10'): 
                avg = round(avg, 0)
            else:
                avg = round(avg, 1) #4204.36 becomes 4204.4
            moving_avg.append(avg)
       
        return moving_avg
    
    
    def calc_avg_aqi(self, gas_num, conc_vals, hours):
        return self.calc_sub_aqi(gas_num, self.hour_avg(hours*6, gas_num, conc_vals), hours)
    
    def sub_aqi(self, gas_name, conc_vals):
        aqi_val = []
        o3_8 = []
        o3_1 = []
        if gas_name == 'CO':
            aqi_val.append(self.calc_avg_aqi(gas_name, conc_vals, 8))
        elif gas_name == 'NO2':
            aqi_val.append(self.calc_avg_aqi(gas_name, conc_vals, 24))
        elif gas_name == 'O3':
            o3_1[:] = self.calc_avg_aqi(gas_name, conc_vals, 1)
            o3_8[:] = self.calc_avg_aqi(gas_name, conc_vals, 8)
            for i in range(7):
                aqi_val.append(o3_1[i])
            for i in range(7,len(o3_8)):
                if o3_8[i] is None or o3_8[i] > 392:
                    aqi_val.append(o3_1[i])
                elif o3_1[i] is not None:
                    aqi_val.append(max(o3_1[i], o3_8[i]))
                else:
                    aqi_val.append(o3_8[i])
            #aqi_val.append(calc_aqi(gas_name, conc_val, 8))
        elif gas_name == 'SO2':
            so2_1 = self.calc_avg_aqi(gas_name, conc_vals, 1)
            so2_24 = self.calc_avg_aqi(gas_name, conc_vals, 24)
            for i in range(23):
                aqi_val.append(so2_1[i])
            for i in range(23, len(so2_1)):
                if so2_1[i] is None or so2_1[i] > 797:
                    aqi_val.append(so2_24[i])
                elif so2_24[i] is not None:
                    aqi_val.append(max(so2_1[i], so2_24[i]))
                else:
                    aqi_val.append(so2_1[i])
        else:
            aqi_val.append(self.calc_avg_aqi(gas_name, conc_vals, 24))
        return aqi_val
    


    def calc_sub_aqi (self, gas_name, conc_vals, hours):
        bp_list=[]
        aqi_vals = []
        found=0
        if gas_name == 'CO':
            bp_list[:] = self.co_8_breakpoint
        elif gas_name == 'NO2' and hours==24:
            bp_list[:] = self.no2_24_breakpoint
        elif gas_name == 'O3' and hours==1:
            bp_list[:] = self.o3_1_breakpoint
        elif gas_name == 'O3' and hours==8:
            bp_list[:] = self.o3_8_breakpoint
        elif gas_name == 'SO2' and hours==1:
            bp_list[:] = self.so2_1_breakpoint
        elif gas_name == 'SO2' and hours==24:
            bp_list[:] = self.so2_24_breakpoint
        
        elif gas_name == 'PM10' and hours==24:
            bp_list[:] = self.pm10_24_breakpoint
        else:
            bp_list[:] = self.pm2_5_24_breakpoint
       
        
        for i in range(len(conc_vals)):
            found = 0
            
            
            for index, bp_tuple in enumerate(bp_list):
                
                if found==1:
                    break
                if len(bp_tuple)==2:
                    bp_low, bp_high = bp_tuple
               
                if  len(bp_tuple)==2 and (conc_vals[i]>=bp_low) and (conc_vals[i]<=bp_high):
                    
                    found=1
                    
                    aqi_val = (((self.aqi_categories[index][1] - self.aqi_categories[index][0])/(bp_high-bp_low))*(conc_vals[i]-bp_low)) + self.aqi_categories[index][0]
                        
                    aqi_vals.append(aqi_val)
                  
                    
            if found==0:
                aqi_vals.append(None)
        
        return aqi_vals
            
    def calc_aqi (self,aqi_vals):
        return max(aqi_vals, key=lambda x: float('-inf') if x is None else x)
            

    def calc_aqi_category(self, aqi_val):
        aqi_list = []
        aqi_list[:] = self.aqi_categories
        for index, aqi_thruple in enumerate(aqi_list):
            aqi_low, aqi_high, aqi_category = aqi_thruple
            if(aqi_val>=aqi_low) and (aqi_val<=aqi_high):
                return aqi_category

