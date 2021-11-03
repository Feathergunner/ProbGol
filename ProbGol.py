#!usr/bin/python
# -*- coding: utf-8 -*-import string

import tkinter as tk
import numpy as np
import time
from functools import partial

COLORS = ['black', 'green', 'red', 'blue', 'yellow', 'cyan', 'purple']

HZSTEPS = [100,200,400,500,1000]

class Population:
	id_counter = 1
	def __init__(self):
		self.id = Population.id_counter
		Population.id_counter += 1
		self.p_ld = np.zeros(9)
		self.p_ll = np.zeros(9)
		self.relevant_other_populations = [self.id]
		
	#def set_rule_ld(self, k, p):
	#	self.p_ld[k] = p
		
	#def set_rule_ll(self, k, p):
	#	self.p_ll[k] = p
		
	def set_relevant_population(self, k):
		if not k in self.relevant_other_populations:
			self.relevant_other_populations.append(k)
	
	def remove_relevant_population(self, k):
		if k in self.relevant_other_populations:
			self.relevant_other_populations.remove(k)

class ProbGol:
	'''
	Args:
		size (int): sqrt of world size
		n (int): number of individual populations
	Attributes:
		current_iter (int): current iteration
		world (2-d np-array of int): the world, always storing the last and the current iteration
		p_ld (np-array of floats): the probabilities of life for dead cells, for each population and each possible number of neighbors
		p_ll (np-array of floats): the probabilities of life for living cells, for each population and each possible number of neighbors
		population_size (np-array of int): the current number of living cells for each population
	'''
	def __init__(self, size, n=1):
		self.size = size
		self.n_pop = n
		self.current_iter = 0
		self.populations = [Population() for _ in range(self.n_pop)]
		self.reset_world()
		
	def add_pop(self):
		self.n_pop += 1
		self.populations.append(Population())
		
	def remove_pop(self):
		self.n_pop -= 1
		self.populations.pop()
		
	### BASIC ITERATION ###
	def next_iteration(self):
		# set iteration counter:
		self.current_iter += 1
		# reset popuation sizes:
		self.population_size = np.zeros(self.n_pop+1).astype(int)
		
		# get new population:
		for x in range(self.size):
			for y in range(self.size):
				c = np.random.random()
				currentpop = self.world[x,y, (self.current_iter-1)%2]
				
				# get numbers of neighbors:
				num_neighbors = [0 for _ in range(self.n_pop+1)]
				for i in range(-1,2):
					for j in range(-1,2):
						if not (i==0 and j==0):
							num_neighbors[self.world[(x+i)%self.size, (y+j)%self.size, (self.current_iter-1)%2]] += 1
				
				# p(life|death):
				if currentpop == 0:
					pop = 0
					newlife = False
					while not newlife and pop < self.n_pop:
						pop += 1
						num_neighbors_pop = sum([num_neighbors[k] for k in self.populations[pop-1].relevant_other_populations])
						if c < self.populations[pop-1].p_ld[num_neighbors_pop]:
							self.world[x,y, self.current_iter%2] = pop
							self.population_size[pop] += 1
							newlife = True
					if not newlife:
						self.world[x,y, self.current_iter%2] = 0
				#p(life|life):
				else:
					num_neighbors_pop = sum([num_neighbors[k] for k in self.populations[currentpop-1].relevant_other_populations])
					if c < self.populations[currentpop-1].p_ll[num_neighbors_pop]:
						self.world[x,y, self.current_iter%2] = currentpop
						self.population_size[currentpop] += 1
					else:
						self.world[x,y, self.current_iter%2] = 0
		
	### WORLD CONTROL ###
	def generate_random_world(self):
		self.world[:,:,0] = np.random.randint(self.n_pop+1, size=(self.size, self.size))
		self.world[:,:,1] = np.random.randint(self.n_pop+1, size=(self.size, self.size))
	
	def reset_world(self):
		self.world = np.zeros((self.size, self.size, 2)).astype(int)
		self.population_size = np.zeros(self.n_pop+1).astype(int)
		
	def get_worldsize(self):
		return self.size**2

class ProgGolView(tk.Frame):
	def __init__(self, master, probgol, hz_id=0):
		super().__init__(master)
		self.master = master
		self.hz_id = hz_id
		self.probgol = probgol
		
		self.timer = HZSTEPS[self.hz_id]
		self.pop_hist = np.zeros((probgol.n_pop, 100))
		self.popmeter_line_ids = [[] for _ in range(probgol.n_pop)]
	
		self._init_world_canvas()
		self.draw_world()
		self._init_window()
		
	### INITIALIZATION ###
	def _init_window(self):
		self.frame = tk.Frame(self.master)
		self.frame.grid(row=0, column=1)
		
		# simulation parameters:
		self.frame_sim = tk.Frame(self.frame, borderwidth=5, relief="groove")
		self.frame_sim.grid(row=0, column=1)
		self.text_label_currentspeed = tk.StringVar(self.frame_sim, value='Simulation speed: '+str(1000/self.timer)+' FPS ')
		self.label_currentspeed = tk.Label(self.frame_sim, textvar=self.text_label_currentspeed).grid(row=0, column=0)
		self.text_label_pause = tk.StringVar(self.frame_sim, value='')
		self.label_pause = tk.Label(self.frame_sim, textvar=self.text_label_pause).grid(row=1, column=0)
		
		self.frame_sim_buttons = tk.Frame(self.frame_sim, borderwidth=2, relief="sunken")
		self.frame_sim_buttons.grid(row=2, column=0)
		button_speed_dec = tk.Button(self.frame_sim_buttons, text='-', command=self.decrease_speed).grid(row=0, column=0)
		button_speed_pause = tk.Button(self.frame_sim_buttons, text='||', command=self.pause_sim).grid(row=0, column=1)
		button_speed_inc = tk.Button(self.frame_sim_buttons, text='+', command=self.increase_speed).grid(row=0, column=2)
		
		# ProbGol parameters:
		self.frame_params = tk.Frame(self.frame, borderwidth=5, relief="groove")
		self.frame_params.grid(row=1, column=1)
		
		self.p_ld_entries = []
		self.p_ll_entries = []
		for k in range(self.probgol.n_pop):
			self._probgol_frame_parameter_input(self.frame_params, k)
		
		# Population density tracking:
		self.pm_h_scale = 2
		self.pm_w_scale = 5
		self.pm_h = 100*self.pm_h_scale
		self.pm_w = 100*self.pm_w_scale
		self.canvas_popmeter = tk.Canvas(self.frame, bg="black", height=self.pm_h, width=self.pm_w)
		self.canvas_popmeter.create_line(0, 25*self.pm_h_scale, self.pm_w, 25*self.pm_h_scale, width=1, fill='gray')
		self.canvas_popmeter.create_line(0, 50*self.pm_h_scale, self.pm_w, 50*self.pm_h_scale, width=1, fill='gray')
		self.canvas_popmeter.create_line(0, 75*self.pm_h_scale, self.pm_w, 75*self.pm_h_scale, width=1, fill='gray')
		self.canvas_popmeter.grid(row=2, column=1)
		self.pop_label_text = tk.StringVar(self.frame, value='Population: 0')
		self.pop_label = tk.Label(self.frame, textvar=self.pop_label_text).grid(row=3, column=1)
		button_reset_world = tk.Button(self.frame, text="Reset world", command=self.probgol.reset_world).grid(row=4, column=1)
		button_rand_world = tk.Button(self.frame, text="Randomize world", command=self.probgol.generate_random_world).grid(row=5, column=1)
		
		self.pause = False
		self.after(self.timer, self.step)
		
	def _probgol_frame_parameter_input(self, parent, k):
		self.frame_params_k = tk.Frame(parent, borderwidth=2, relief="sunken")
		self.frame_params_k.grid(row=1, column=k+1)
		
		l_p_top = tk.Label(self.frame_params_k, text="P(life|death)").grid(row=0,column=1, columnspan=3)
		l_p_top = tk.Label(self.frame_params_k, text="P(life|life)").grid(row=0,column=4, columnspan=3)
		self.frames_ld = []
		self.p_ld_entries_k = []
		self.p_ll_entries_k = []
		for i in range(9):
			tk.Label(self.frame_params_k, text=str(i)).grid(row=i+1, column=0)
			v_ld = tk.StringVar(self.frame_params_k, value=str(self.probgol.populations[0].p_ld[i]))
			self.p_ld_entries_k.append(v_ld)
			tk.Entry(self.frame_params_k, width=5, justify=tk.RIGHT, textvariable=v_ld).grid(row=i+1, column=1)
			tk.Button(self.frame_params_k, text='+', command=partial(self.increase_ld, k,i)).grid(row=i+1, column=2)
			tk.Button(self.frame_params_k, text='-', command=partial(self.decrease_ld, k,i)).grid(row=i+1, column=3)
			v_ll = tk.StringVar(self.frame_params_k, value=str(self.probgol.populations[0].p_ll[i]))
			self.p_ll_entries_k.append(v_ll)
			tk.Entry(self.frame_params_k, width=5, justify=tk.RIGHT, textvariable=v_ll).grid(row=i+1, column=4)
			tk.Button(self.frame_params_k, text='+', command=partial(self.increase_ll, k,i)).grid(row=i+1, column=5)
			tk.Button(self.frame_params_k, text='-', command=partial(self.decrease_ll, k,i)).grid(row=i+1, column=6)
			
		self.p_ld_entries.append(self.p_ld_entries_k)
		self.p_ll_entries.append(self.p_ll_entries_k)
		
		
	### SIMULATION CONTROL ###
	def increase_speed(self):
		if self.hz_id > 0:
			self.hz_id -= 1
			self.timer = HZSTEPS[self.hz_id]
		self.text_label_currentspeed.set('Simulation speed: '+str(1000/self.timer)+' FPS ')
			
	def decrease_speed(self):
		if self.hz_id < len(HZSTEPS)-1:
			self.hz_id += 1
			self.timer = HZSTEPS[self.hz_id]
		self.text_label_currentspeed.set('Simulation speed: '+str(1000/self.timer)+' FPS ')
		
	def pause_sim(self):
		if not self.pause:
			self.pause=True
			self.text_label_pause.set("(PAUSED)")
		else:
			self.pause=False
			self.text_label_pause.set("")
			self.step()
			
	
	### CANVAS CONTROL ###
	def _init_world_canvas(self):
		cellsize = 800//self.probgol.size
		self.canvas_world = tk.Canvas(self.master, bg="white", height=cellsize*self.probgol.size+20, width=cellsize*self.probgol.size+20)
		self.canvas_world.grid(row=0, column=0)
		self.cell_ids = []
		for x in range(self.probgol.size):
			self.cell_ids.append([])
			for y in range(self.probgol.size):
				self.cell_ids[x].append(self.canvas_world.create_rectangle(10+x*cellsize, 10+y*cellsize, 10+(x+1)*cellsize, 10+(y+1)*cellsize, fill='black'))

	def draw_world(self):
		for i in range(self.probgol.size):
			for j in range(self.probgol.size):
				self._draw_cell(i,j)

	def _draw_cell(self, i,j):
		self.canvas_world.itemconfig(self.cell_ids[i][j], fill=COLORS[self.probgol.world[i,j, self.probgol.current_iter%2]])

	def _draw_popmeter(self):
		if self.probgol.current_iter < 100:
			for k in range(self.probgol.n_pop):
				self.popmeter_line_ids[k].append(
					self.canvas_popmeter.create_line(
					(self.probgol.current_iter-1)*self.pm_w_scale,
					self.pm_h - 100*self.pop_hist[k, self.probgol.current_iter-1]//self.probgol.get_worldsize()*self.pm_h_scale,
					self.probgol.current_iter*self.pm_w_scale,
					self.pm_h - 100*self.pop_hist[k, self.probgol.current_iter]//self.probgol.get_worldsize()*self.pm_h_scale,
					width=3, fill=COLORS[k+1]))
		else:
			for k in range(self.probgol.n_pop):
				for i in range(99):
					self.canvas_popmeter.coords(self.popmeter_line_ids[k][i],
						i*self.pm_w_scale,
						self.pm_h - 100*self.pop_hist[k, (i+self.probgol.current_iter)%100]//self.probgol.get_worldsize()*self.pm_h_scale,
						(i+1)*self.pm_w_scale,
						self.pm_h - 100*self.pop_hist[k, (i+1+self.probgol.current_iter)%100]//self.probgol.get_worldsize()*self.pm_h_scale)

	### ITERATION & INPUT CONTROL ###
	def step(self):
		ts = time.time()
		# get state of world:
		self.probgol.next_iteration()
		# draw world:
		self.draw_world()
		
		# get current life-probabilities:
		for k in range(self.probgol.n_pop):
			for i in range(9):
				try:
					self.probgol.populations[k].p_ld[i] = float(self.p_ld_entries[k][i].get())
					self._update_p_ld(k,i)
				except ValueError:
					pass
				try:
					self.probgol.populations[k].p_ll[i] = float(self.p_ll_entries[k][i].get())
					self._update_p_ll(k,i)
				except ValueError:
					pass
		
		for k in range(1, self.probgol.n_pop+1):
			self.pop_hist[k-1, self.probgol.current_iter%100] = self.probgol.population_size[k]
			
		# draw popmeter:
		self._draw_popmeter()
		
		full_pop_label_text = ""
		for k in range(1, self.probgol.n_pop+1):
			full_pop_label_text += "Population "+str(k)+": "+str(100*self.probgol.population_size[k]//self.probgol.get_worldsize())#+"%\n"
		self.pop_label_text.set(full_pop_label_text)
		
		t_delta = time.time() - ts
		wait = int(max(0, self.timer-t_delta))
		if not self.pause:
			self.after(wait, self.step)
			
	def _get_probabilities_from_input(self):
		for k in range(self.probgol.n_pop):
			for i in range(9):
				try:
					self.probgol.populations[k].p_ld[i] = float(self.p_ld_entries[k][i].get())
					self._update_p_ld(k,i)
				except ValueError:
					pass
				try:
					self.probgol.populations[k].p_ll[i] = float(self.p_ll_entries[k][i].get())
					self._update_p_ll(k,i)
				except ValueError:
					pass
			
	### INPUT CONTROL ###
	# TODO
	def _update_p_ld(self,k,i):
		if self.probgol.populations[k].p_ld[i] < 0:
			self.probgol.populations[k].p_ld[i] = 0
		elif self.probgol.populations[k].p_ld[i] > 1:
			self.probgol.populations[k].p_ld[i] = 1
		self.probgol.populations[k].p_ld[i] = round(self.probgol.populations[k].p_ld[i],2)
		self.p_ld_entries[k][i].set("{:0<.2f}".format(self.probgol.populations[k].p_ld[i]))
	
	def _update_p_ll(self,k,i):
		if self.probgol.populations[k].p_ll[i] < 0:
			self.probgol.populations[k].p_ll[i] = 0
		elif self.probgol.populations[k].p_ll[i] > 1:
			self.probgol.populations[k].p_ll[i] = 1
		self.probgol.populations[k].p_ll[i] = round(self.probgol.populations[k].p_ll[i],2)
		self.p_ll_entries[k][i].set("{:0<.2f}".format(self.probgol.populations[k].p_ll[i]))
		
	def increase_ld(self, k, i):
		self.probgol.populations[k].p_ld[i] += 0.05
		self._update_p_ld(k,i)
		
	def decrease_ld(self, k, i):
		self.probgol.populations[k].p_ld[i] -= 0.05
		self._update_p_ld(k,i)
		
	def increase_ll(self, k, i):
		self.probgol.populations[k].p_ll[i] += 0.05
		self._update_p_ll(k,i)
		
	def decrease_ll(self, k, i):
		self.probgol.populations[k].p_ll[i] -= 0.05
		self._update_p_ll(k,i)

probgol = ProbGol(size=50,n=2)
app = ProgGolView(master=tk.Tk(), probgol=probgol)
app.mainloop()